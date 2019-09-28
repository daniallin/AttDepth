import cv2
import json
import torch
import os.path
import numpy as np
from torchvision import transforms
from dataload.transform_img import Compose, RandomHorizontalFlip, RandomCenterCrop,\
    RandomRotate, ColorJitter, RandomVerticalFlip

DEPTH_PARAM_1 = 351.3
DEPTH_PARAM_2 = 1092.5


class NYUV2Raw():
    def __init__(self, file_path, args, phase='train'):
        self.args = args
        self.phase = phase
        self.dir_anno = os.path.join(file_path, phase + '_raw_annotations.json')
        self.rgb_paths, self.depth_paths = self.get_paths()
        self.uniform_size = (480, 640)

    def get_paths(self):
        with open(self.dir_anno, 'r') as load_f:
            paths = json.load(load_f)
        self.data_size = len(paths)

        rgb_paths = [paths[i]['rgb_path'] for i in range(len(paths))]
        depth_path = [paths[i]['depth_path'] for i in range(len(paths))]

        return rgb_paths, depth_path

    def __getitem__(self, index):
        data = self.get_data(index)
        return data

    def get_data(self, index):
        rgb_path = self.rgb_paths[index]
        depth_path = self.depth_paths[index]

        with open(rgb_path, 'rb') as r:
            rgb_raw = self.read_ppm(r)  # bgr, H*W*C
        with open(depth_path, 'rb') as d:
            depth_rel = self.read_pgm(d)

        # Projects a depth image from internal Kinect coordinates to world coordinates.
        depth_abs = DEPTH_PARAM_1 / (DEPTH_PARAM_2 - depth_rel)
        depth_raw = np.clip(depth_abs, 0, None)

        rgb, depth = self.img_transformer([rgb_raw, depth_raw])

        rgb = rgb.transpose((2, 0, 1))  # H*W*C to C*H*W
        depth = depth[np.newaxis, :, :]

        # change the color channel, bgr -> rgb
        rgb = rgb[::-1, :, :]

        # to torch, normalize
        rgb = self.scale_torch(rgb, 255.)
        depth = self.scale_torch(depth, 1)

        data = {'rgb': rgb, 'depth': depth}
        return data

    def img_transformer(self, imgs):
        clr = ColorJitter()
        if self.phase == 'train':
            compose = Compose((RandomCenterCrop((self.args.crop_size[0], self.args.crop_size[1])),
                               RandomHorizontalFlip(),
                               RandomVerticalFlip(),
                               RandomRotate()))

            imgs[0] = clr(imgs[0])
            imgs = compose(imgs)
        elif self.phase == 'val':
            compose = Compose((RandomCenterCrop((self.args.crop_size[0], self.args.crop_size[1])),))
            imgs[0] = clr(imgs[0])
            imgs = compose(imgs)
        elif self.phase == 'test':
            imgs = imgs

        return imgs

    def scale_torch(self, img, scale):
        img = img.astype(np.float32)
        img /= scale
        img = torch.from_numpy(img.copy())
        if img.size(0) == 3:
            img = transforms.Normalize(self.args.img_mean, self.args.img_std)(img)
        return img

    def __len__(self):
        return self.data_size

    def read_pgm(self, pgm_file):
        """Reads a PGM file from a buffer.

        Returns a numpy array of the appropiate size and dtype.
        """

        # First line contains some image meta-info
        p5, width, height, depth = pgm_file.readline().split()

        # Ensure we're actually reading a P5 file
        assert p5 == b'P5'
        assert depth == b'65535', "Only 16-bit PGM files are supported"

        width, height = int(width), int(height)

        data = np.fromfile(pgm_file, dtype='<u2', count=width * height)

        return data.reshape(height, width).astype(np.uint32)

    def read_ppm(self, ppm_file):
        """Reads a PPM file from a buffer.

        Returns a numpy array of the appropiate size and dtype.
        """

        p6, width, height, depth = ppm_file.readline().split()

        assert p6 == b'P6'
        assert depth == b'255', "Only 8-bit PPM files are supported"

        width, height = int(width), int(height)

        data = np.fromfile(ppm_file, dtype=np.uint8, count=width * height * 3)

        return data.reshape(height, width, 3)




