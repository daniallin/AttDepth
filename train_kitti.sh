#!/usr/bin/env bash
python train_att.py use_cuda True --epochs 20 --batch-size 2 --dataset kitti --kitti_path dataset/kitti --min_depth 0.1 --max_depth 80 --crop_size 480, 960
