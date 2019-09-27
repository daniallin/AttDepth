import os
import time
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
# import torchvision

from utils.params import set_params
from utils.helper import set_random_seed, AverageMeter
from utils.keeper import Keeper
from utils.loss import ssim, grad_loss
from models import build_model
from dataload import data_loader


def main(args):
    model = build_model(args.model_name, args)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

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

        optimizer.load_state_dict(checkpoint['optimizer'])
        best_loss = checkpoint['best_loss']
        args.start_epoch = checkpoint['epoch'] + 1
    else:
        best_loss = np.inf

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

        model.train()

        for k, (train_img, train_depth) in enumerate(tqdm(train_loader)):
            # print(train_img.size())
            if k > 0: break
            train_depth = train_depth.type(torch.FloatTensor)
            if args.use_cuda:
                train_img, train_depth = train_img.cuda(), train_depth.cuda()

            optimizer.zero_grad()

            train_pred = model(train_img)
            rmse = torch.sqrt(mse_criterion(train_pred, train_depth))
            l_depth = l1_criterion(train_pred, train_depth)
            l_edge = grad_loss(train_pred, train_depth)
            l_ssim = torch.clamp((1 - ssim(train_pred, train_depth, val_range=1000.0 / 10.0)) * 0.5, 0, 1)
            # print("losses: ", rmse, l_edge, l_ssim)

            train_loss = (10 * l_ssim) + (10 * l_edge) + (1.0 * l_depth)
            train_loss.backward()
            optimizer.step()
            writer.add_scalar('training loss', train_loss, epoch * train_bts + k)
            writer.add_scalar('training RMSE', rmse, epoch * train_bts + k)

            log.info('train  combine loss and rmse of epoch/batch {}/{} are {} and {}'.format(epoch, k, train_loss, rmse))
            keeper.save_loss([epoch, train_loss.item(), rmse.item(), l_depth.item(),
                              l_edge.item(), l_ssim.item()], 'train_losses.csv')

        # evaluating test data
        val_avg = AverageMeter()
        model.eval()
        with torch.no_grad():
            for k, (val_img, val_depth) in enumerate(tqdm(val_loader)):
                if k > 0: break
                val_depth = val_depth.type(torch.FloatTensor)
                if args.use_cuda:
                    val_img, val_depth = val_img.cuda(), val_depth.cuda()

                val_pred = model(val_img)

                if k % 20 == 1:
                    keeper.save_img([val_img[0], val_depth[0], val_pred[0]])

                val_loss = torch.sqrt(mse_criterion(val_pred, val_depth))
                val_avg.update(val_loss.item())
                log.info('val rmse of epoch/batch {}/{} is {}'.format(epoch, k, val_loss))

                writer.add_scalar('validation RMSE', val_loss, epoch * train_bts + k)

            keeper.save_loss([val_avg.avg], 'val_losses.csv')

        if val_avg.avg < best_loss:
            best_loss = val_avg.avg
            keeper.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
            }, 'best_model.pth')

        log.info('Saving model ...')
        keeper.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,
        })

        log.info('training time of epoch {}/{} is {} \n'.format(epoch + 1, args.epochs, time.time() - e_time))


if __name__ == '__main__':
    set_random_seed()
    args = set_params()
    args.depth_bin_interval = (np.log10(args.max_depth) - np.log10(args.min_depth)) / args.decoder_output_c

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

