import cv2
import json
import torch
import os.path
import numpy as np
import scipy.io as sio
import h5py
import torchvision.transforms as transforms


class HitachiDataset():
    def __init__(self, dataroot, args, phase='train'):
        self.dataroot = dataroot
        self.args = args
        self.phase = phase
        self.dir_anno = os.path.join(dataroot, phase + '_annotations.json')
        self.A_paths, self.B_paths, self.AB_anno = self.getData()
        self.data_size = len(self.AB_anno)
        self.uniform_size = (720, 1280)

    def getData(self):
        with open(self.dir_anno, 'r') as load_f:
            AB_anno = json.load(load_f)

        A_list = [AB_anno[i]['rgb_path'] for i in range(len(AB_anno))]
        B_list = [AB_anno[i]['depth_path'] for i in range(len(AB_anno))]
        return A_list, B_list, AB_anno

    def __getitem__(self, anno_index):
        data = self.online_aug(anno_index)
        return data

    def online_aug(self, anno_index):
        A_path = self.A_paths[anno_index]
        B_path = self.B_paths[anno_index]
        # print('='*20, B_path)

        A = cv2.imread(A_path)  # bgr, H*W*C
        B = cv2.imread(B_path, -1) / 1000.0  # the max depth is 20m

        flip_flg, crop_size, pad, resize_ratio = self.set_flip_pad_reshape_crop()

        A_resize = self.flip_pad_reshape_crop(A, flip_flg, crop_size, pad, 128)
        B_resize = self.flip_pad_reshape_crop(B, flip_flg, crop_size, pad, -1)

        A_resize = A_resize.transpose((2, 0, 1))
        B_resize = B_resize[np.newaxis, :, :]

        # change the color channel, bgr -> rgb
        A_resize = A_resize[::-1, :, :]

        # to torch, normalize
        A_resize = self.scale_torch(A_resize, 255.)
        B_resize = self.scale_torch(B_resize, 1)

        # return (A_resize, B_resize)

        data = {'A': A_resize, 'B': B_resize}
        return data

    def set_flip_pad_reshape_crop(self):
        """
        Set flip, padding, reshaping, and cropping factors for the image.
        :return:
        """
        # flip
        flip_prob = np.random.uniform(0.0, 1.0)
        flip_flg = True if flip_prob > 0.5 and 'train' in self.phase else False

        raw_size = np.array([self.args.crop_size[1], 416, 448, 480, 512, 544, 576, 608, 640])
        size_index = np.random.randint(0, 9) if 'train' in self.phase else 8

        # pad
        pad_height = raw_size[size_index] - self.uniform_size[0] if raw_size[size_index] > self.uniform_size[0]\
                    else 0
        pad = [pad_height, 0, 0, 0]  # [up, down, left, right]

        # crop
        crop_height = raw_size[size_index]
        crop_width = raw_size[size_index]
        start_x = np.random.randint(0, int(self.uniform_size[1] - crop_width)+1)
        start_y = 0 if pad_height != 0 else np.random.randint(0,
                int(self.uniform_size[0] - crop_height) + 1)
        crop_size = [start_x, start_y, crop_height, crop_width]

        resize_ratio = float(self.args.crop_size[1] / crop_width)

        return flip_flg, crop_size, pad, resize_ratio

    def flip_pad_reshape_crop(self, img, flip, crop_size, pad, pad_value=0):
        """
        Flip, pad, reshape, and crop the image.
        :param img: input image, [C, H, W]
        :param flip: flip flag
        :param crop_size: crop size for the image, [x, y, width, height]
        :param pad: pad the image, [up, down, left, right]
        :param pad_value: padding value
        :return:
        """
        # Flip
        if flip:
            img = np.flip(img, axis=1)

        # Pad the raw image
        if len(img.shape) == 3:
            img_pad = np.pad(img, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0)), 'constant',
                             constant_values=(pad_value, pad_value))
        else:
            img_pad = np.pad(img, ((pad[0], pad[1]), (pad[2], pad[3])), 'constant',
                             constant_values=(pad_value, pad_value))
        # Crop the resized image
        img_crop = img_pad[crop_size[1]:crop_size[1] + crop_size[3], crop_size[0]:crop_size[0] + crop_size[2]]

        # Resize the raw image
        img_resize = cv2.resize(img_crop, (self.args.crop_size[1], self.args.crop_size[0]), interpolation=cv2.INTER_LINEAR)
        return img_resize

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
        return img

    def __len__(self):
        return self.data_size

