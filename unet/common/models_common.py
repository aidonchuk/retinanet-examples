import torch
from torch import nn
from torchvision.models import resnet18


class ResNet18(nn.Module):

    def __init__(self, num_classes=1, num_filters=32, pretrained=True, is_deconv=True, requires_grad=True):
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = resnet18(pretrained=pretrained)

        for params in self.encoder.parameters():
            params.requires_grad = requires_grad

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool
                                   )

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlock(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlock(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        x_out = self.final(dec0)
        return x_out


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=False, one_conv=False):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:

            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels,
                                   kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            if not one_conv:
                self.block = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    ConvRelu(in_channels, middle_channels),
                    ConvRelu(middle_channels, out_channels),
                )
            else:
                self.block = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    ConvRelu(in_channels, out_channels),
                )

    def forward(self, x):
        return self.block(x)


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


def conv1x1(in_, out):
    return nn.Conv2d(in_, out, 1, padding=0)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class ConvRelu1x1(nn.Module):
    def __init__(self, in_: int, out: int):
        super(ConvRelu1x1, self).__init__()
        self.conv = conv1x1(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
