import os
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from utils.params import set_params
from utils.helper import AverageMeter, time_lr_scheduler
from utils.keeper import Keeper
from utils.loss import ssim, grad_loss, depth_scale_invariant
from models import build_model
from dataload import data_loader


def main(args):
    model = build_model(args.model_name, args)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    log.info('loading data...\n')
    train_loader, val_loader = data_loader(args)
    train_bts, val_bts = len(train_loader), len(val_loader)
    log.info('train batch number: {0}; validation batch number: {1}'.format(train_bts, val_bts))

    # Whether using checkpoint
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise RuntimeError("=> no checkpoint found")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()

        optimizer.load_state_dict(checkpoint['optimizer'])
        best_loss = checkpoint['best_loss']
        args.start_epoch = checkpoint['epoch'] + 1
        start_step = checkpoint['step'] + 1
    else:
        best_loss = np.inf
        start_step = 0

    # whether using pretrained model
    if args.pretrained_net is not None and args.resume is None:
        pretrained_w = torch.load(args.pretrained_net)
        model_dict = model.state_dict()
        pretrained_dict = {k: torch.from_numpy(v) for k, v in pretrained_w.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # run on multiple GPUs by DataParallel
    model = nn.DataParallel(model, device_ids=[int(e) for e in args.gpu_ids]) if args.use_cuda else model
    l1_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()

    # -------------------- training -------------------- #
    for epoch in range(args.start_epoch, args.epochs):
        e_time = time.time()
        log.info('training: epoch {}/{} \n'.format(epoch+1, args.epochs))
        scheduler.step()

        model.train()
        for k, train_data in enumerate(tqdm(train_loader)):
            # print(train_img.size())
            # if k > 10: break
            train_depth = train_data['depth']
            train_img = train_data['rgb']
            if args.use_cuda:
                train_img, train_depth = train_img.cuda(), train_depth.cuda()

            optimizer.zero_grad()

            train_pred, loss_sigma = model(train_img)

            # remove depth out of max depth
            mask = train_depth.le(args.max_depth) & train_depth.ge(args.min_depth)
            mask = mask.type(torch.FloatTensor)
            if args.use_cuda:
                mask = mask.cuda()
            train_depth *= mask
            train_pred *= mask

            rmse = torch.sqrt(mse_criterion(train_pred, train_depth))
            # l_depth = l1_criterion(train_pred, train_depth)
            l_depth = depth_scale_invariant(train_pred, train_depth)
            l_edge = grad_loss(train_pred, train_depth)
            l_ssim = torch.sqrt((1 - ssim(train_pred, train_depth, val_range=args.max_depth / args.min_depth)))
            # print("losses: ", rmse, l_edge, l_ssim)

            # train_loss = (10 * l_ssim) + (10 * l_edge) + (1.0 * l_depth)
            # train_loss = sum(1 / (2 * torch.exp(loss_sigma[i])) * loss + loss_sigma[i] / 2 for i in range(3) for loss in [l_depth, l_edge, l_ssim])
            # train_loss = sum(loss_sigma[i] * loss for i in range(3) for loss in [l_depth, l_edge, l_ssim])
            train_loss = sum([l_depth, l_edge, l_ssim])
            train_loss.backward()
            optimizer.step()
            writer.add_scalar('training loss', train_loss, epoch * train_bts + k)
            # writer.add_scalar('training RMSE', rmse, epoch * train_bts + k)

            log.info('train  combine loss and rmse of epoch/batch {}/{} are {} and {}'.format(epoch, k, train_loss, rmse))
            keeper.save_loss([epoch, train_loss.item(), rmse.item(), l_depth.item(),
                              l_edge.item(), l_ssim.item()], 'train_losses.csv')

            if k % 500 == 0:
                writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch * train_bts + k)

        # evaluating test data
        val_rmse_avg = AverageMeter()
        val_relabs_avg = AverageMeter()
        model.eval()
        with torch.no_grad():
            for k, val_data in enumerate(tqdm(val_loader)):
                # if k > 10: break
                val_depth = val_data['depth']
                val_img = val_data['rgb']
                if args.use_cuda:
                    val_img, val_depth = val_img.cuda(), val_depth.cuda()

                val_pred, _ = model(val_img)

                # remove depth out of max depth
                mask = val_depth.le(args.max_depth) & val_depth.ge(args.min_depth)
                mask = mask.type(torch.FloatTensor)
                if args.use_cuda:
                    mask = mask.cuda()
                val_depth *= mask
                val_pred *= mask

                if k % 200 == 1:
                    keeper.save_img(epoch, k, [val_img[0], val_depth[0], val_pred[0]])

                val_rmse = torch.sqrt(mse_criterion(val_pred, val_depth))
                val_depth[val_depth.eq(0)] = 1e-5
                val_relabs = torch.mean(torch.abs(val_pred - val_depth) / val_depth)
                val_rmse_avg.update(val_rmse.item())
                val_relabs_avg.update(val_relabs.item())
                log.info('val rmse of epoch/batch {}/{} is {}'.format(epoch, k, val_rmse))

                writer.add_scalar('validation RMSE', val_rmse, epoch * val_bts + k)
                writer.add_scalar('validation relative absolute error', val_relabs, epoch * val_bts + k)
            writer.add_scalar('Epoch validation RMSE', val_rmse_avg.avg, epoch)
            writer.add_scalar('Epoch validation absErrorRel', val_relabs_avg.avg, epoch)

            # keeper.save_loss([val_rmse_avg.avg], 'val_losses.csv')

            # optimizer = time_lr_scheduler(optimizer, epoch, lr_decay_epoch=3)

        log.info('Saving model ...')
        keeper.save_checkpoint({
            'epoch': epoch,
            'step': 0,
            # 'state_dict': model.state_dict(),  # cpu
            'state_dict': model.module.state_dict(),  # GPU
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,
        }, is_best=val_rmse_avg.avg < best_loss)

        if val_rmse_avg.avg < best_loss:
            best_loss = val_rmse_avg.avg

        log.info('training time of epoch {}/{} is {} \n'.format(epoch + 1, args.epochs, time.time() - e_time))

        start_step = 0


if __name__ == '__main__':
    # set_random_seed()
    args = set_params()

    keeper = Keeper(args)
    log = keeper.setup_logger()
    writer = SummaryWriter(os.path.join(keeper.experiment_dir, 'tb_run'),
                           comment='LR_{}_BATCH_{}'.format(args.lr, args.batch_size))
    log.info('Welcome to summoner\'s rift')

    cuda_exist = torch.cuda.is_available()
    if args.use_cuda and cuda_exist:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in args.gpu_ids)
        print("Let's use GPU(s): {}. Current Device: {}".format(
            torch.cuda.device_count(), torch.cuda.current_device()))

    log.info(args)
    keeper.save_experiment_config()

    start_time = time.time()

    log.info("Thirty seconds until minion spawn!")

    main(args)

    writer.close()

    log.info('Victory! Total game time is: {}'.format(time.time()-start_time))

