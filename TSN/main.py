import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import logging
import logging.config
import random
import numpy as np

from torch.nn.utils import clip_grad_norm

from dataset import TSNDataSet
from models import TSN
from transforms import *
from opts import parser
from tqdm import tqdm, trange
from sys import stdout

#logger = logging.getLogger(__name__)
#formatter = logging.Formatter('%(asctime)s - %(name)s -%(levelname)s - %(message)s')
#handler = logging.StreamHandler(stdout)
#handler.setFormatter(formatter)
#logger.addHandler(handler)
#logger.setLevel(logging.INFO)

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #logging.basicConfig(level=logging.DEBUG,
    #                    format="%(asctime)s %(name)-12s % (levelname)-8s %(message)s",
    #                    datefmt="%m-%d %H:%M",
    #                    filename=args.output_dir + "model_results.log",
    #                    filemode="w")
    #logging.config.fileConfig(args.output_dir + "model_result.log")
    logging.basicConfig(filename=args.output_dir + "model_result.log")

    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 600
    elif args.dataset == 'activitynet':
        num_class = 200
    else:
        raise ValueError('Unknown dataset '+args.dataset)

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn)
    if args.pre_model:
        print("=> loading pre-trained model '{}'".format(args.pre_model))
        state_dict = torch.load(args.pre_model, map_location='cpu')
        model.base_model.load_state_dict(state_dict, strict=False)

    print (model)
    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    else:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    for epoch in trange(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader))
            #print ("prec1", prec1)
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s - %(name)s -%(levelname)s - %(message)s')
    handler = logging.StreamHandler(stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    #logging.config.fileConfig(args.output_dir + "model_result.log")

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()

    # set train_acc
    train_acc_top1, train_acc_top5 = [], []

    for i, (input, target) in tqdm(enumerate(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        acc_top1 = compute_acc(output.data, target, topk=1)
        acc_top5 = compute_acc(output.data, target, topk=5)
        train_acc_top1.append(acc_top1)
        train_acc_top5.append(acc_top5)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #if i % args.print_freq == 0:
        print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))
    print ("training {} epoch acc top1 is {}".format(epoch, np.mean(train_acc_top1)))
    print ("training {} epoch acc top5 is {}".format(epoch, np.mean(train_acc_top5)))

    #logging.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
    #              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
    #              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
    #              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
    #               epoch, i, len(train_loader), batch_time=batch_time,
    #               data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))
    acc_top1_log = np.mean(train_acc_top1)
    acc_top5_log = np.mean(train_acc_top5)
    logger.info("training {0} epoch acc top1 is {1}".format(epoch, acc_top1_log))

    logger.info("training {0} epoch acc top5 is {1}".format(epoch, acc_top5_log))

def validate(val_loader, model, criterion, iter, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s - %(name)s -%(levelname)s - %(message)s')
    handler = logging.StreamHandler(stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    #logging.config.fileConfig(args.output_dir + "model_result.log")

    # switch to evaluate mode
    model.eval()

    end = time.time()

    # set test_acc
    test_acc_top1, test_acc_top5 = [], []

    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        acc_top1 = compute_acc(output.data, target, topk=1)
        acc_top5 = compute_acc(output.data, target, topk=5)
        test_acc_top1.append(acc_top1)
        test_acc_top5.append(acc_top5)

        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #if i % args.print_freq == 0:
        #    print(('Test: [{0}/{1}]\t'
        #          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #           i, len(val_loader), batch_time=batch_time, loss=losses,
        #           top1=top1, top5=top5)))

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses)))

    logger.info("test acc_top1 is {}".format(np.mean(test_acc_top1)))
    logger.info("test acc_top5 is {}".format(np.mean(test_acc_top5)))
    print ("test acc_top1 is {}".format(np.mean(test_acc_top1)))
    print ("test acc_top5 is {}".format(np.mean(test_acc_top5)))

    #logging.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
    #      .format(top1=top1, top5=top5, loss=losses)))
    #test_acc_top1_log = np.mean(test_acc_top1)
    #test_acc_top5_log = np.mean(test_acc_top5)
    #print (test_acc_top1_log)
    #if acc_top1_log is None:
    #    acc_top1_log, acc_top5_log = 0, 0
    #logger.info("test acc_top1 is {}".format(test_acc_top1_log))
    #logger.info("test acc_top5 is {}".format(test_acc_top5_log))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, args.modality.lower(), filename))
    filename = args.output_dir + filename
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)
        torch.save(state, best_name)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def compute_acc(output, target, topk=1):
    '''compute the acc@k for the specified values of k'''
    batch_size = target.size(0)

    _, output = output.topk(topk)
    same_num = 0
    for tgt, out in zip(target, output):
        if tgt in out:
            same_num += 1

    return same_num * 100 / batch_size


if __name__ == '__main__':
    #logging.basicConfig(level=logging.INFO,
    #                    format="%(asctime)s %(name)-12s % (levelname)-8s %(message)s",
    #                    datefmt="%m-%d %H:%M",
    #                    filename=args.output_dir + "model_results.log",
    #                    filemode="w")
    main()

