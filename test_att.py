import os
import time
import cv2
from tqdm import tqdm
import numpy as np
import json

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from models import build_model
from utils.params import set_params


class HitachiDataset():
    def __init__(self, dataroot):
        self.dataroot = dataroot
        self.dir_anno = os.path.join(dataroot, 'test.json')
        self.A_paths = self.getData()
        self.uniform_size = (720, 1280)

    def getData(self):
        with open(self.dir_anno, 'r') as load_f:
            A_list = json.load(load_f)

        A_list = [A_list[i]['rgb_path'] for i in range(len(A_list))]
        return A_list

    def __getitem__(self, index):
        data = self.online_aug(index)
        return data

    def online_aug(self, index):
        A_path = self.A_paths[index]

        A = cv2.imread(A_path)  # bgr, H*W*C

        A = A.transpose((2, 0, 1))

        # change the color channel, bgr -> rgb
        A = A[::-1, :, :]

        # to torch, normalize
        A = self.scale_torch(A, 255.)
        return A, A_path

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
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
        return img


def main(use_cuda, json_path, pretrained_model, args):
    data_set = HitachiDataset(json_path)
    test_loader = DataLoader(data_set, 1, shuffle=False)

    model = build_model(args.model_name, args)

    # Whether using checkpoint
    checkpoint = torch.load(pretrained_model)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.cuda()

    model = nn.DataParallel(model, device_ids=[int(e) for e in args.gpu_ids]) if args.use_cuda else model
    model.eval()
    with torch.no_grad():
        for k, (val_img, img_path) in enumerate(tqdm(test_loader)):
            # if k > 0: break
            if use_cuda:
                val_img = val_img.cuda()

            pred_depth = model(val_img)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            pred_depth_scale = (pred_depth * 1000)

            save_name = 'Depth_' + img_path.split('/')[-1].split('_')[-1].replace('jpg', 'png')
            cv2.imwrite(os.path.join('result', save_name), pred_depth_scale, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join('0')

    args = set_params()
    args.depth_bin_interval = (np.log10(args.max_depth) - np.log10(args.min_depth)) / args.decoder_output_c

    pretrained = 'pretrained/best_model.pth'
    json_path = 'dataset/hitachi'
    if not os.path.exists(json_path):
        raise NotImplementedError

    main(use_cuda, json_path, pretrained, args)
