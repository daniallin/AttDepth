import argparse
import numpy as np


def set_params():
    parser = argparse.ArgumentParser(description='Depth Estimation using Monocular Images')
    parser.add_argument('--dataset', type=str, default='kitti', choices=['nyuv2', 'kitti', 'nyu_raw'])
    parser.add_argument('--nyu_path', type=str, default='dataset/nyuv2')
    parser.add_argument('--kitti_path', type=str, default='dataset/kitti')
    parser.add_argument('--do_kb_crop', type=bool, default=True, help='Cropping images as kitti benchmark images')
    parser.add_argument('--img_mean', default=(0.485, 0.456, 0.406), type=tuple)
    parser.add_argument('--img_std', default=(0.229, 0.224, 0.225), type=tuple)
    parser.add_argument('--min_depth', type=float, default=0.1, help='Minimum of input depths, 0.001 for nyu; 0.1 for kitti')
    parser.add_argument('--max_depth', type=float, default=80, help='Maximum of input depths, 10 for nyu; 80 for kitti')
    parser.add_argument('--min_depth_log', type=float, default=np.log(0.001), help='Minimum depth in log space')
    parser.add_argument('--minus_point_5', default=False, type=bool)
    parser.add_argument('--crop_size', type=float, default=(320, 480), help='height & width, 480*960 for kitti; 384*480 fot nyu')

    # train
    # parser.add_argument('--num_workers', type=int, default=8, help='number workers using for dataloader')
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--gpu_ids', type=str, default=['0'], help='IDs of GPUs to use')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sync_bn', type=bool, default=False, help='whether using sync batch normalization')
    parser.add_argument('--output_scale', type=int, default=32, help='output scale of encoder')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='w-decay (default: 5e-4)')

    # model
    parser.add_argument('--freeze_bn', type=bool, default=False)
    parser.add_argument('--backbone', type=str, default='resnext', choices=['resnext', 'sknet'], help='encoder model')
    parser.add_argument('--model_name', type=str, default='AttDepth')
    parser.add_argument('--pretrained_net', type=str, default=None)
    parser.add_argument('--use_pretrain', type=bool, default=True, help='whether using pretrained encoder network')
    parser.add_argument('--resume', type=str, default=None, help='Start training from an existing model.')
    parser.add_argument('--save_path', type=str, default='train_results/')

    args = parser.parse_args()
    return args


