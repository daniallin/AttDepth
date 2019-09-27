import torch
import torch.nn as nn
import torch.nn.functional as F

from models.AttDepth.resnext import resnext101_32x8d
from models.AttDepth.aspp import ASPP
from models.AttDepth.multi_decoder import Decoder


class AttDepth(nn.Module):
    def __init__(self, args):
        super(AttDepth, self).__init__()
        self.encoder = resnext101_32x8d(args.use_pretrain, replace_stride_with_dilation=[False, False, True])
        self.aspp = ASPP(args)
        self.decoder = Decoder(args)

        if args.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def forward(self, input):
        # input: batch_size, channel, height, width
        output, low_level_feature = self.encoder(input)
        output = self.aspp(output)
        output = self.decoder(output, low_level_feature)
        # print(output[0].size())

        return output


if __name__ == '__main__':
    from utils.params import set_params
    args = set_params()
    model = AttDepth(args)
    model.eval()
    input = torch.rand(1, 3, 320, 480)
    output = model(input)
    print(output.size())




