import os
from pathlib import Path
import math
import json
import numpy as np
import random
from PIL import Image
import torch
from torchvision import transforms


def get_rgb_depth(raw_path, anno_path):
    for p in os.listdir(anno_path):
        phase = os.path.join(anno_path, p)
        img_path = list()
        for ind in os.listdir(phase):
            rgb_depth = dict()
            date = ind[:10]
            depth_path = os.path.join(phase, ind, 'proj_depth/groundtruth/image_02')
            rgb_path = os.path.join(raw_path, date, ind, 'image_02/data')
            if not os.path.exists(rgb_path):
                continue
            depth_list = os.listdir(depth_path)
            rgb_list = os.listdir(rgb_path)
            for d in depth_list:
                if d in rgb_list:
                    rgb_depth['rgb_path'] = Path(os.path.join(rgb_path, d)).as_posix()
                    rgb_depth['depth_path'] = Path(os.path.join(depth_path, d)).as_posix()
                    img_path.append(rgb_depth)
        random.shuffle(img_path)
        with open("../dataset/kitti/{}_raw_annotations.json".format(p), "w") as f:
            json.dump(img_path, f)


def calculate_rgb_mean_std(img_path_list, minus_point_5=False):
    n_images = len(img_path_list)
    cnt_pixels = 0
    print('Numbers of frames in training dataset: {}'.format(n_images))
    mean_np = [0, 0, 0]
    mean_tensor = [0, 0, 0]
    to_tensor = transforms.ToTensor()

    image_sequence = []
    for idx, img_path in enumerate(img_path_list):
        print('{} / {}'.format(idx, n_images), end='\r')
        img_as_img = Image.open(img_path)
        img_as_tensor = to_tensor(img_as_img)
        if minus_point_5:
            img_as_tensor = img_as_tensor - 0.5
        img_as_np = np.array(img_as_img)
        img_as_np = np.rollaxis(img_as_np, 2, 0)
        cnt_pixels += img_as_np.shape[1]*img_as_np.shape[2]
        for c in range(3):
            mean_tensor[c] += float(torch.sum(img_as_tensor[c]))
            mean_np[c] += float(np.sum(img_as_np[c]))
    mean_tensor =  [v / cnt_pixels for v in mean_tensor]
    mean_np = [v / cnt_pixels for v in mean_np]
    print('mean_tensor = ', mean_tensor)
    print('mean_np = ', mean_np)

    std_tensor = [0, 0, 0]
    std_np = [0, 0, 0]
    for idx, img_path in enumerate(img_path_list):
        print('{} / {}'.format(idx, n_images), end='\r')
        img_as_img = Image.open(img_path)
        img_as_tensor = to_tensor(img_as_img)
        if minus_point_5:
            img_as_tensor = img_as_tensor - 0.5
        img_as_np = np.array(img_as_img)
        img_as_np = np.rollaxis(img_as_np, 2, 0)
        for c in range(3):
            tmp = (img_as_tensor[c] - mean_tensor[c])**2
            std_tensor[c] += float(torch.sum(tmp))
            tmp = (img_as_np[c] - mean_np[c])**2
            std_np[c] += float(np.sum(tmp))
    std_tensor = [math.sqrt(v / cnt_pixels) for v in std_tensor]
    std_np = [math.sqrt(v / cnt_pixels) for v in std_np]
    print('std_tensor = ', std_tensor)
    print('std_np = ', std_np)


if __name__ == '__main__':
    # must set the path as your virtual kitti dataset path
    get_rgb_depth('E:/DepthEstimation/AttDepth/dataset/kitti_raw', 'E:/Datasets_Win/KITTI/depth/data_depth_annotated')

    # img_path_list = []
    # for p in rgb:
    #     img_path_list.extend(glob.glob(p + '/*.png'))
    # calculate_rgb_mean_std(img_path_list, minus_point_5=args.minus_point_5)
