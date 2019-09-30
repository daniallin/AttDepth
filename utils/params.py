import argparse
import numpy as np


def set_params():
    parser = argparse.ArgumentParser(description='Depth Estimation using Monocular Images')
    parser.add_argument('--dataset', type=str, default='nyu_raw', choices=['nyuv2', 'kitti', 'nyu_raw'])
    parser.add_argument('--nyu_path', type=str, default='dataset/nyuv2')
    parser.add_argument('--kitti_path', type=str, default='dataset/kitti/')
    parser.add_argument('--img_mean', default=(0.485, 0.456, 0.406), type=tuple)
    parser.add_argument('--img_std', default=(0.229, 0.224, 0.225), type=tuple)
    parser.add_argument('--min_depth', type=float, default=0.001, help='Minimum of input depths')
    parser.add_argument('--max_depth', type=float, default=10, help='Maximum of input depths')
    parser.add_argument('--min_depth_log', type=float, default=np.log(0.001), help='Minimum depth in log space')
    parser.add_argument('--depth_bin_interval', type=int, default=None, help='interval of each bin')
    parser.add_argument('--depth_bin_border', type=int, default=None, help='boundary of each bin')
    parser.add_argument('--minus_point_5', default=False, type=bool)
    parser.add_argument('--crop_size', type=float, default=(384, 480), help='height & width')

    # train
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--gpu_ids', type=str, default=['0'], help='IDs of GPUs to use')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--sync_bn', type=bool, default=False, help='whether using sync batch normalization')
    parser.add_argument('--output_scale', type=int, default=16, help='output scale of encoder')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='w-decay (default: 5e-4)')

    # model
    parser.add_argument('--freeze_bn', type=bool, default=False)
    parser.add_argument('--backbone', type=str, default='resnext', choices=['resnext', 'sknet'], help='encoder model')
    parser.add_argument('--decoder_output_c', type=int, default=150, help='output channels of decoder')
    parser.add_argument('--model_name', type=str, default='AttDepth', choices=['DORN', 'BTS', 'AttDepth'])
    parser.add_argument('--pretrained_net', type=str, default=None)
    parser.add_argument('--use_pretrain', type=bool, default=True, help='whether using pretrained encoder network')
    parser.add_argument('--resume', type=str, default=None, help='Start training from an existing model.')
    parser.add_argument('--save_path', type=str, default='train_results/')

    args = parser.parse_args()
    return args


