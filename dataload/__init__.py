from torch.utils.data import DataLoader

from dataload.nyuv2_labeled import NYUV2Dataset
from dataload.nyuv2_raw import NYUV2Raw


def data_loader(args, **kwargs):
    if args.dataset == 'nyuv2':
        train_set = NYUV2Dataset(args.nyu_path, args)
        val_set = NYUV2Dataset(args.nyu_path, args, phase='val')
        print('Train_size: {}. Validation size: {}'.format(len(train_set), len(val_set)))

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, val_loader

    elif args.dataset == 'nyu_raw':
        train_set = NYUV2Raw(args.nyu_path, args)
        val_set = NYUV2Raw(args.nyu_path, args, phase='val')
        print('Train_size: {}. Validation size: {}'.format(len(train_set), len(val_set)))

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, val_loader
    else:
        raise NotImplementedError


if __name__ == '__main__':
    from utils.params import set_params
    import torch
    import matplotlib.pyplot as plt


    def plot_color(ax, color, title="Color"):
        """Displays a color image from the NYU dataset."""

        ax.axis('off')
        ax.set_title(title)
        ax.imshow(color)

    args = set_params()
    train_loader, val_loader = data_loader(args)
    for k, train_data in enumerate(train_loader):
        # print(train_img.size())
        if k > 0: break
        train_depth = train_data['depth'].type(torch.FloatTensor)
        train_img = train_data['rgb']

        fig = plt.figure("Raw Dataset Sample", figsize=(12, 5))

        ax = fig.add_subplot(1, 2, 1)
        plot_color(ax, train_depth[0].mean(dim=0).numpy(), "depth")

        ax = fig.add_subplot(1, 2, 2)
        plot_color(ax, train_img[0].numpy().transpose((1, 2, 0)), "img")

        plt.show()
