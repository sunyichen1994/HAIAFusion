import torch

from models.CMDAF import CMDAF
from models.common import reflect_conv
from torch import nn


def Fusion(vi_out, ir_out):
    return torch.cat([vi_out, ir_out], dim=1)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.vi_conv1 = nn.Conv2d(in_channels=1, kernel_size=3, out_channels=16, stride=1, padding=1)
        self.ir_conv1 = nn.Conv2d(in_channels=1, kernel_size=3, out_channels=16, stride=1, padding=1)

        self.vi_conv2 = reflect_conv(in_channels=16, kernel_size=3, out_channels=16, stride=1, pad=1)
        self.ir_conv2 = reflect_conv(in_channels=16, kernel_size=3, out_channels=16, stride=1, pad=1)

        self.vi_conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.ir_conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)

        self.vi_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.ir_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)

        self.vi_conv5 = reflect_conv(in_channels=64, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.ir_conv5 = reflect_conv(in_channels=64, kernel_size=3, out_channels=128, stride=1, pad=1)

    def forward(self, y_vi_image, ir_image):
        activate = nn.PReLU()
        activate = activate.cuda()
        vi_out = activate(self.vi_conv1(y_vi_image))
        ir_out = activate(self.ir_conv1(ir_image))

        vi_out, ir_out = CMDAF(activate(self.vi_conv2(vi_out)), activate(self.ir_conv2(ir_out)))
        vi_out, ir_out = CMDAF(activate(self.vi_conv3(vi_out)), activate(self.ir_conv3(ir_out)))
        vi_out, ir_out = CMDAF(activate(self.vi_conv4(vi_out)), activate(self.ir_conv4(ir_out)))

        vi_out, ir_out = activate(self.vi_conv5(vi_out)), activate(self.ir_conv5(ir_out))
        return vi_out, ir_out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = reflect_conv(in_channels=256, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.conv2 = reflect_conv(in_channels=128, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.conv3 = reflect_conv(in_channels=64, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=16, stride=1, pad=1)
        self.conv5 = nn.Conv2d(in_channels=16, kernel_size=1, out_channels=1, stride=1, padding=0)

    def forward(self, x):
        activate = nn.PReLU()
        activate = activate.cuda()
        x = activate(self.conv1(x))
        x = activate(self.conv2(x))
        x = activate(self.conv3(x))
        x = activate(self.conv4(x))
        x = nn.Tanh()(self.conv5(x)) / 2 + 0.5
        return x


class HAIAFusion(nn.Module):
    def __init__(self):
        super(HAIAFusion, self).__init__()
        self.generator = Encoder()
        self.discriminator = Discriminator()

    def forward(self, y_vi_image, ir_image):
        vi_generator_out, ir_generator_out = self.generator(y_vi_image, ir_image)
        generator_out = Fusion(vi_generator_out, ir_generator_out)
        fused_image = self.discriminator(generator_out)
        return fused_image
