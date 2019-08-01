import os
import io
import argparse
import pandas as pd
import numpy as np
import zipfile
import torch
import cv2
import json
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from models import TSN


class ZipDataset(Dataset):
    def __init__(self, video_path, modality='RGB', snippet=16, transform=None):
        self.modality = modality
        self.transform = transform
        if modality == 'RGB':
            cap = cv2.VideoCapture(video_path)
            wid = int(cap.get(3))
            hei = int(cap.get(4))
            framerate = int(cap.get(5))
            framenum = int(cap.get(7))
            print(wid, hei, framerate, framenum) 
            self.video = np.zeros((framenum,hei,wid,3),dtype='uint8')
            cnt = 0
            while(cap.isOpened()):
                a,b=cap.read()
                if b is None:
                    break
                b = b.astype('uint8')
                self.video[cnt]=b
                cnt+=1
            self.img_list = []
            video_name = video_path.split('/')[-1].split('.')[0]
            video_duration = framenum*1./framerate
            video_featureFrame = int(framenum/16)*16
            df = pd.DataFrame({'video':[video_name], 
                  "numFrame":[framenum], 
                  "seconds":[video_duration], 
                  "fps":[framerate], 
                  "rfps": [video_featureFrame/video_duration], 
                  "subset":["validation"], 
                  "featureFrame":[video_featureFrame]})
            df.to_csv('video_info.csv', )
            video_anno_all = {}
            video_anno = {}
            video_anno["duration_second"] = video_duration
            video_anno["duration_frame"] = framenum
            video_anno["annotations"] = []
            video_anno["feature_frame"] = video_featureFrame
            video_anno_all[video_name] = video_anno
            with open('video_anno.json', 'w') as f:
                json.dump(video_anno_all, f)
            
            for i in range(0, framenum, snippet):
                self.img_list.append(i)

    def __getitem__(self, idx):
        if self.modality == 'RGB':
            img = self.video[idx]
            img = Image.fromarray(np.uint8(img))
            if self.transform:
                img = self.transform(img)
            return img

        else:
            raise NotImplementedError

    def __len__(self):
        if self.modality == 'RGB':
            return len(self.img_list)
        else:
            raise NotImplementedError

    def __del__(self):
        try:
            self.zip_fp.close()
        except:
            pass
        try:
            self.flow_zip_fp[0].close()
        except:
            pass
        try:
            self.flow_zip_fp[1].close()
        except:
            pass


def inference(model, dataloader):
    logits = []
    for batch in dataloader:
        batch = torch.autograd.Variable(batch, volatile=True).cuda()
        out = model(batch).data.cpu().numpy().copy()
        logits.append(out)
    logits = np.concatenate(logits, axis=0)
    return logits


class Roll(object):
    def __call__(self, tensor):
        inv_idx = torch.arange(2, -1, -1).long()
        return tensor[inv_idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path')
    parser.add_argument('--weights', '-w', default='./dpn92_RGB_k600_fold_2_rgb_model_best.pth.tar')
    parser.add_argument('--save_dir', '-s', default='output_ftrs')
    parser.add_argument('--modality', '-m', default='RGB', choices=['RGB', 'Flow', 'RGBDiff'])
    parser.add_argument('--arch', default="dpn92")
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--num_workers', '-j', type=int, default=4)
    args = parser.parse_args()

#     df = pd.read_csv(args.list_file, sep=' ', names=['path'], usecols=[0])
#     df.drop_duplicates('path', inplace=True)

    net = TSN(200, 1, args.modality, base_model=args.arch, consensus_type='identity', dropout=0.5)

    if args.arch == 'BNInception':
        input_mean = [x / 255. for x in net.input_mean]
        transform = T.Compose([
            T.Scale(net.scale_size, Image.BILINEAR),
            T.CenterCrop(net.input_size),
            T.ToTensor(),
            Roll(),
            T.Normalize(input_mean, net.input_std)
        ])
    elif 'resnet' in args.arch or 'resnext' in args.arch:
        transform = T.Compose([
            T.Scale(net.scale_size, Image.BILINEAR),
            T.CenterCrop(net.input_size),
            T.ToTensor(),
            T.Normalize(net.input_mean, net.input_std)
        ])
    elif 'dpn' in args.arch:
        transform = T.Compose([
            T.Scale(net.scale_size, Image.BILINEAR),
            T.CenterCrop(net.input_size),
            T.ToTensor(),
            T.Normalize(net.input_mean, net.input_std)
        ])
    elif 'Inception' in args.arch:
        transform = T.Compose([
            T.Scale(net.scale_size, Image.BILINEAR),
            T.CenterCrop(net.input_size),
            T.ToTensor(),
            T.Normalize(net.input_mean, net.input_std)
        ])
    else:
        raise NotImplementedError

    checkpoint = torch.load(args.weights)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}

    net = net.cuda()
    net.load_state_dict(base_dict)
    net.eval()
    save_dir = args.save_dir + '_' + args.modality + '_' + args.arch
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    csv_path = os.path.join(save_dir, os.path.basename(args.video_path).split('.')[0] + '.csv')
#     if os.path.exists(csv_path):
#         continue

    dataloader = DataLoader(
        dataset=ZipDataset(args.video_path, args.modality, 16, transform),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    logits = inference(net, dataloader)
    frange = range(200) if args.modality == 'RGB' else range(200, 400)
    np.savetxt(csv_path, logits, '%.11f', ',', header=','.join(['f{}'.format(i) for i in frange]))
