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
    parser.add_argument('list_file')
    parser.add_argument('weights')
    parser.add_argument('--save_dir', '-s', default='output_ftrs')
    parser.add_argument('--modality', '-m', default='RGB', choices=['RGB', 'Flow', 'RGBDiff'])
    parser.add_argument('--arch', default="se_resnext101_32x4d")
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--num_workers', '-j', type=int, default=4)
    args = parser.parse_args()

    df = pd.read_csv(args.list_file, sep=' ', names=['path'], usecols=[0])
    df.drop_duplicates('path', inplace=True)

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

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        csv_path = os.path.join(save_dir, os.path.basename(row['path']) + '.csv')
        if os.path.exists(csv_path):
            continue

        dataloader = DataLoader(
            dataset=ZipDataset(row['path'], args.modality, 16, transform),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )

        logits = inference(net, dataloader)
        frange = range(200) if args.modality == 'RGB' else range(200, 400)
        np.savetxt(csv_path, logits, '%.11f', ',', header=','.join(['f{}'.format(i) for i in frange]))
