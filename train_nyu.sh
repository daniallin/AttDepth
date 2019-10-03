#!/usr/bin/env bash
python train_att.py use_cuda True --epochs 20 --batch-size 2 --dataset nyu_raw --nyu_path dataset/nyuv2 --min_depth 0.01 --max_depth 10 --crop_size 384, 480
