import os
import io
import argparse
import pandas as pd
import numpy as np
import zipfile
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from models import TSN
#from models_weights import TSN

seed = 42
torch.manual_seed(42)

class ZipDataset(Dataset):
    def __init__(self, dir_path, modality='RGB', snippet=16, transform=None):
        self.modality = modality
        self.transform = transform
        if modality == 'RGB':
            self.img_zip_path = os.path.join(dir_path, 'img.zip')
            self.img_list = []
            self.img_zip_fp = zipfile.ZipFile(self.img_zip_path)
            num_frames = len(self.img_zip_fp.namelist())
            for i in range(0, num_frames, snippet):
                self.img_list.append('img_{:0>5}.jpg'.format(i + 1))

        elif modality == 'Flow':
            self.flow_zip_paths = [os.path.join(dir_path, 'flow_x.zip'), os.path.join(dir_path, 'flow_y.zip')]
            self.flow_x_list = []
            self.flow_y_list = []
            self.flow_zip_fp = [zipfile.ZipFile(p) for p in self.flow_zip_paths]
            num_frames = len(self.flow_zip_fp[0].namelist())
            for i in range(0, num_frames, snippet):
                snippet_flow_x = []
                snippet_flow_y = []
                for _ in range(5):
                    snippet_flow_x.append('x_{:0>5}.jpg'.format(i + 1))
                    snippet_flow_y.append('y_{:0>5}.jpg'.format(i + 1))
                    if i < num_frames - 1:
                        i += 1
                self.flow_x_list.append(snippet_flow_x)
                self.flow_y_list.append(snippet_flow_y)

    def __getitem__(self, idx):
        if self.modality == 'RGB':
            try:
                with self.zip_fp.open(self.img_list[idx]) as f:
                    img = Image.open(f)
            except:
                self.zip_fp = zipfile.ZipFile(self.img_zip_path)
                with self.zip_fp.open(self.img_list[idx]) as f:
                    img = Image.open(f)

            if self.transform:
                img = self.transform(img)
            return img

        elif self.modality == 'Flow':
            flow_x_img = []
            flow_y_img = []

            for name in self.flow_x_list[idx]:
                try:
                    with self.flow_zip_fp[0].open(name) as f:
                        flow_x_img.append(self.transform(Image.open(f)))
                except:
                    self.flow_zip_fp[0] = zipfile.ZipFile(self.flow_zip_paths[0])
                    with self.flow_zip_fp[0].open(name) as f:
                        flow_x_img.append(self.transform(Image.open(f)))

            for name in self.flow_y_list[idx]:
                try:
                    with self.flow_zip_fp[1].open(name) as f:
                        flow_y_img.append(self.transform(Image.open(f)))
                except:
                    self.flow_zip_fp[1] = zipfile.ZipFile(self.flow_zip_paths[1])
                    with self.flow_zip_fp[1].open(name) as f:
                        flow_y_img.append(self.transform(Image.open(f)))

            flow_imgs = []
            for x, y in zip(flow_x_img, flow_y_img):
                flow_imgs.extend([x, y])
            return torch.cat(flow_imgs, dim=0)

        else:
            raise NotImplementedError

    def __len__(self):
        if self.modality == 'RGB':
            return len(self.img_list)
        elif self.modality == 'Flow':
            return len(self.flow_x_list)
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

def batch_padding(batch):
    padding = 20 - len(batch)
    if padding == 0:
        return batch, padding
    else:
        #image_padding = [Image.fromarray(np.zeros((224, 224, 3)).astype(np.uint8)) for _ in range(padding)]
        #image_result = [i for j in [batch, image_padding] for i in j]
        image_padding = torch.zeros((padding, 3, 224, 224))
        image_result = torch.cat((batch, image_padding), dim=0)
        return image_result, padding

def inference(model, dataloader):
    logits = []
    for batch in dataloader:
        #print ("batch.size", batch.size())
        #batch, padding = batch_padding(batch)
        #print ("batch.size", len(batch))
        #print ("image.size", batch[0])
        batch = torch.autograd.Variable(batch, volatile=True).cuda()
        #print ("batch.size", batch.size())
        #_, _, cls = model(batch)
        out = model(batch)
        #print ("out.size", out.size())
        #print ("weights.size", weights.size())
        out = out.data.cpu().numpy()
        #weights = weights.data.cpu().numpy()
        #print ("output.data", _.data)
        #start = start.data.cpu().numpy()
        #print ('start.data', start)
        #end = end.data.cpu().numpy()
        #cls = cls.data.cpu().numpy()
        #print ("cls.data", cls)
        #out = out.reshape(-1, out.shape[-1])
        #print ("out.size", out.shape)
        #start = start.reshape(-1, start.shape[-1])
        #end = end.reshape(-1, end.shape[-1])
        #cls = cls.reshape(-1, cls.shape[-1])
        #cls = cls[:len(cls)-padding]
        #print ("start.size", start.shape)
        #print ("end.size", end.shape)
        #print ("cls.size", cls.shape)
        out = out.reshape(-1, out.shape[-1])
        #weights = weights.reshape(-1, weights.shape[-1])
        #out = np.concatenate([out, weights], axis=1)
        #print ("out.size", out.shape)
        logits.append(out)
    logits = np.concatenate(logits, axis=0)
    return logits


class Roll(object):
    def __call__(self, tensor):
        inv_idx = torch.arange(2, -1, -1).long()
        return tensor[inv_idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_file')
    parser.add_argument('--weights')
    parser.add_argument('--save_dir', '-s', default='output_ftrs')
    parser.add_argument('--modality', '-m', default='RGB', choices=['RGB', 'Flow', 'RGBDiff'])
    parser.add_argument('--arch', default="se_resnext101_32x4d")
    parser.add_argument('--batch_size', '-b', type=int, default=256)
    parser.add_argument('--num_workers', '-j', type=int, default=16)
    parser.add_argument('--num_snippet', type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.list_file, sep=' ', names=['path'], usecols=[0])
    df.drop_duplicates('path', inplace=True)

    net = TSN(200, 1, args.modality, base_model=args.arch, consensus_type='identity', dropout=0.8)

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
    elif 'inception' in args.arch:
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
    #print (net)
    net.load_state_dict(base_dict)
    net.eval()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        csv_path = os.path.join(args.save_dir, os.path.basename(row['path']) + '.csv')
        if os.path.exists(csv_path):
            continue

        dataloader = DataLoader(
            dataset=ZipDataset(row['path'], args.modality, args.num_snippet, transform),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )

        logits = inference(net, dataloader)
        frange = range(200+128) if args.modality == 'RGB' else range(200, 400)
        np.savetxt(csv_path, logits, '%.11f', ',', header=','.join(['f{}'.format(i) for i in frange]))
