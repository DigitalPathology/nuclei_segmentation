import torch
import torch.nn as nn



def double_conv_down(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=out_channels),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=out_channels)
    )

def double_conv_up(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):

    def __init__(self, n_class, in_channels, filterCountInFirstLayer, param_1, param_2, param_3, param_4, param_6, param_8, param_12, param_16, param_24):
        super().__init__()

        self.dconv_down1 = double_conv_down(in_channels=in_channels, out_channels=filterCountInFirstLayer)
        self.dconv_down2 = double_conv_down(in_channels=filterCountInFirstLayer, out_channels=param_2 * filterCountInFirstLayer)
        self.dconv_down3 = double_conv_down(in_channels=param_2 * filterCountInFirstLayer, out_channels=param_4 * filterCountInFirstLayer)
        self.dconv_down4 = double_conv_down(in_channels=param_4 * filterCountInFirstLayer, out_channels=param_8 * filterCountInFirstLayer)
        self.dconv_down5 = double_conv_down(in_channels=param_8 * filterCountInFirstLayer, out_channels=param_16 * filterCountInFirstLayer)

        #self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = double_conv_up(in_channels=param_24 * filterCountInFirstLayer, out_channels=param_8 * filterCountInFirstLayer)
        self.dconv_up3 = double_conv_up(in_channels=param_12 * filterCountInFirstLayer, out_channels=param_4 * filterCountInFirstLayer)
        self.dconv_up2 = double_conv_up(in_channels=param_6 * filterCountInFirstLayer , out_channels=param_2 * filterCountInFirstLayer)
        self.dconv_up1 = double_conv_up(in_channels=param_3 * filterCountInFirstLayer , out_channels=param_1 * filterCountInFirstLayer)

        self.conv_last = nn.Conv2d(in_channels=filterCountInFirstLayer, out_channels=n_class, kernel_size=1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = nn.functional.avg_pool2d(input=conv1, kernel_size=2)

        conv2 = self.dconv_down2(x)
        x = nn.functional.avg_pool2d(input=conv2, kernel_size=2)

        conv3 = self.dconv_down3(x)
        x = nn.functional.avg_pool2d(input=conv3, kernel_size=2)

        conv4 = self.dconv_down4(x)
        x = nn.functional.avg_pool2d(input=conv4, kernel_size=2)

        x = self.dconv_down5(x)

        x = self.upsample(x)
        x = torch.cat(tensors=[x, conv4], dim=1)

        x = self.dconv_up4(x)
        x = self.upsample(x)
        x = torch.cat(tensors=[x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat(tensors=[x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat(tensors=[x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out