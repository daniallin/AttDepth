import cv2
import json
import torch
import os.path
import numpy as np
import torchvision.transforms as transforms
from dataload.transform_img import Compose, RandomHorizontalFlip, RandomCenterCrop,\
    RandomRotate, ColorJitter, RandomResizedCrop


class KITTIDataset():
    def __init__(self, file_path, args, phase='train'):
        self.args = args
        self.phase = phase
        self.dir_anno = os.path.join(file_path, phase + '_raw_annotations.json')
        self.rgb_paths, self.depth_paths = self.get_paths()
        self.depth_normalize = 255.

    def get_paths(self):
        with open(self.dir_anno, 'r') as load_f:
            paths = json.load(load_f)
        self.data_size = len(paths)
        rgb_paths = [paths[i]['rgb_path'] for i in range(len(paths))]
        depth_paths = [paths[i]['depth_path'] for i in range(len(paths))]
        return rgb_paths, depth_paths

    def __getitem__(self, index):
        data = self.get_data(index)
        return data

    def get_data(self, index):
        rgb_path = self.rgb_paths[index]
        depth_path = self.depth_paths[index]
        # print(rgb_path, depth_path)

        rgb_raw = cv2.imread(rgb_path)  # [H, W, C] C:bgr
        depth_raw = cv2.imread(depth_path, -1) / self.depth_normalize

        if self.args.do_kb_crop is True:
            # print('Cropping training images as kitti benchmark images')
            height, width = depth_raw.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            depth_raw = depth_raw[top_margin:top_margin + 352, left_margin:left_margin + 1216]
            rgb_raw = rgb_raw[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

        rgb, depth = self.img_transformer([rgb_raw, depth_raw])

        rgb = rgb.transpose((2, 0, 1))  # H*W*C to C*H*W
        depth = depth[np.newaxis, :, :]

        # change the color channel, bgr -> rgb
        rgb = rgb[::-1, :, :]

        # to torch, normalize
        ratio_list = [1.0, 1.2, 1.5, 1.8, 2.0]  # depth zoom ratio
        resize_ratio = ratio_list[np.random.randint(len(ratio_list))] if self.phase == 'train' else 1.0
        rgb = self.scale_torch(rgb, 255.)
        depth = self.scale_torch(depth, resize_ratio)

        data = {'rgb': rgb, 'depth': depth}
        return data

    def img_transformer(self, imgs):
        # clr = ColorJitter()
        if self.phase == 'train':
            # change from RandomCenterCrop to RandomResizedCrop
            compose = Compose((RandomResizedCrop((self.args.crop_size[0], self.args.crop_size[1])),
                               RandomHorizontalFlip(),
                               RandomRotate()))

            # imgs[0] = clr(imgs[0])
            imgs = compose(imgs)
        elif self.phase == 'val':
            compose = Compose((RandomResizedCrop((self.args.crop_size[0], self.args.crop_size[1])),))
            # imgs[0] = clr(imgs[0])
            imgs = compose(imgs)
        elif self.phase == 'test':
            imgs = imgs

        return imgs

    def scale_torch(self, img, scale):
        """
        Scale the image and output it in torch.tensor.
        :param img: input image. [C, H, W]
        :param scale: the scale factor. float
        :return: img. [C, H, W
        """
        img = img.astype(np.float32)
        img /= scale
        img = torch.from_numpy(img.copy())
        if img.size(0) == 3:
            img = transforms.Normalize(self.args.img_mean, self.args.img_std)(img)
        else:
            img = transforms.Normalize((0,), (1,))(img)
        return img

    def __len__(self):
        return self.data_size

    def name(self):
        return 'KITTI'
