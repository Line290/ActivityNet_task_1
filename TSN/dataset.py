import torch.utils.data as data

from PIL import Image
import os
import os.path
import zipfile
import numpy as np
from numpy.random import randint

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

    def _load_images(self, record, idx):
        directory = os.path.join(self.root_path, record.path)
        images = []
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            with zipfile.ZipFile(os.path.join(directory, 'img.zip')) as zf:
                for i in idx:
                    with zf.open('img_{:0>5}.jpg'.format(i)) as f:
                        images.append(Image.open(f).convert('RGB'))
        elif self.modality == 'Flow':
            with zipfile.ZipFile(os.path.join(directory, 'flow_x.zip')) as zfx, zipfile.ZipFile(os.path.join(directory, 'flow_y.zip')) as zfy:
                for i in idx:
                    for _ in range(self.new_length):
                        with zfx.open('x_{:0>5}.jpg'.format(i)) as f:
                            images.append(Image.open(f).convert('L'))
                        with zfy.open('y_{:0>5}.jpg'.format(i)) as f:
                            images.append(Image.open(f).convert('L'))
                        if i < record.num_frames:
                            i += 1
        return images

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]
        #self.video_list = self.video_list[:5]

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return (offsets + 1).astype(np.int)

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return (offsets + 1).astype(np.int)

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return (offsets + 1).astype(np.int)

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = self._load_images(record, indices)
        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
