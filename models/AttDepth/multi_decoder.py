import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.AttDepth.base import initial_weight


def _conv_layer(in_chans, out_chans, k, s=1, p=1, sync_bn=False):
    conv_block = nn.Sequential(
        nn.Conv2d(in_chans, out_chans, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_chans),
        nn.ReLU(inplace=True), )
    return conv_block


class AttConcat(nn.Module):
    def __init__(self, in_chans, out_chans, method):
        super(AttConcat, self).__init__()
        self.method = method
        # Define layers
        if self.method == 'general':
            self.attention = nn.Conv2d(in_chans, out_chans, 3, padding=1)
        elif self.method == 'concat':
            self.attention = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_chans, out_chans, 3, padding=1))

    def forward(self, x1, x2):
        energy = self._score(x1, x2)
        att_x2 = torch.mul(x2, energy)
        att_x = torch.cat((x1, att_x2), dim=1)
        return att_x

    def _score(self, x1, x2):
        """Calculate the relevance of a particular encoder output in respect to the decoder hidden."""

        if self.method == 'general':
            energy = self.attention(x2)
            energy = torch.bmm(x1.unsqueeze(1), energy.unsqueeze(2))
        elif self.method == 'concat':
            energy = self.attention(torch.cat((x1, x2), dim=1))
        else:
            energy = torch.bmm(x1.unsqueeze(1), x2.unsqueeze(2))
        return F.softmax(energy, dim=1)


class DepthUpSample(nn.Module):
    def __init__(self, in_chans, out_chans, low_size):
        super(DepthUpSample, self).__init__()
        self.attention = AttConcat(in_chans+low_size, low_size, method='concat')
        self.conv = nn.Sequential(nn.Conv2d(in_chans+low_size, out_chans, 3, padding=1),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv2d(out_chans, out_chans, 3, padding=1),
                                  nn.LeakyReLU(0.2))

    def forward(self, x, low_feature):
        up_x = F.interpolate(x, size=low_feature.size()[2:], mode='bilinear', align_corners=True)
        # att_x = torch.mul(up_x, self.attention(up_x, low_feature))
        # att_low = torch.mul(low_feature, self.attention(up_x, low_feature))
        # att_x = torch.cat((up_x, att_low), dim=1)
        att_x = self.attention(up_x, low_feature)
        output = self.conv(att_x)
        return output


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        if args.backbone == 'resnext':
            self.low_feature_sizes = [512, 256, 64]
            self.num_channels = 256
        else:
            raise NotImplementedError

        chan = int(self.num_channels)
        # depth estimation
        self.conv1 = nn.Conv2d(chan, chan, kernel_size=1, stride=1, padding=1)
        self.depth_up1 = DepthUpSample(chan // 1, chan // 2, self.low_feature_sizes[0])
        self.depth_up2 = DepthUpSample(chan // 2, chan // 4, self.low_feature_sizes[1])
        self.depth_up3 = DepthUpSample(chan // 4, chan // 8, self.low_feature_sizes[2])
        self.depth_up4 = nn.Sequential(nn.Conv2d(chan // 8, chan // 16, 3, padding=1),
                                       nn.LeakyReLU(0.2))
        self.last_conv = nn.Sequential(nn.Conv2d(chan // 16, 1, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2))

        initial_weight(self.modules())

    def forward(self, x, low_level_features):
        depth = self.conv1(x)
        depth = self.depth_up1(depth, low_level_features[0])
        depth = self.depth_up2(depth, low_level_features[1])
        depth = self.depth_up3(depth, low_level_features[2])
        depth = self.depth_up4(F.interpolate(depth, size=(depth.size()[-2]*2, depth.size()[-1]*2),
                                             mode='bilinear', align_corners=True))
        depth = self.last_conv(depth)

        return depth


if __name__ == '__main__':
    from utils.params import set_params
    args = set_params()
    model = Decoder(args)
    model.eval()
    x = torch.randn(1, 256, 20, 30)
    low0 = torch.randn(1, 64, 160, 240)
    low1 = torch.randn(1, 256, 80, 120)
    low2 = torch.randn(1, 512, 40, 60)
    y = model(x, [low2, low1, low0])
    print(y.size())
