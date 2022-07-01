import torch
import torch.nn as nn
import torch.nn.functional as F


class FirstSecondOrderMLP(nn.Module):
    def __init__(self, channel, reduction=16):
        super(FirstSecondOrderMLP, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # first order
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        # second order
        y2 = self.count_cov_second(x)
        y2 = y2.mean(1)
        y2 = self.fc(y2).view(b, c, 1, 1)

        y = y + y2

        return x * y.expand_as(x)

    def count_cov_second(self, input):
        x = input
        batchSize, dim, h, w = x.data.shape
        M = h * w
        x = x.reshape(batchSize, dim, M)
        # I_hat = (-1. / M / M) * torch.ones(M, M, device=x.device) + (1. / M) * torch.eye(M, M, device=x.device)
        #
        # I_hat = I_hat.view(1, M, M).repeat(batchSize, 1, 1).type(x.dtype)
        # y = x.bmm(I_hat).bmm(x.transpose(1, 2))
        x_mean_band = x.mean(2).view(batchSize, dim, 1).expand(batchSize, dim, M)
        # y = (x - x_mean_band).bmm(x.transpose(1, 2)) / M
        y = (x - x_mean_band).bmm(x.transpose(1, 2)) / M
        return y


class Conv3x1(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, dilation=1):
        super(Conv3x1, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(padding=(0, 0, reflect_padding, reflect_padding))
        self.conv2d = nn.Conv2d(in_dim, out_dim, (kernel_size, 1), stride, dilation=dilation, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class Conv1x3(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, dilation=1):
        super(Conv1x3, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(padding=(reflect_padding, reflect_padding, 0, 0))
        self.conv2d = nn.Conv2d(in_dim, out_dim, (1, kernel_size), stride, dilation=dilation, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, dilation=1):
        super(Conv3x3, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class SAA_AsySymConvResidual(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SAA_AsySymConvResidual, self).__init__()
        # generate spatial attention 1
        self.asy1_Conv3_1 = Conv3x1(inchannel, inchannel//ratio, 3)
        self.act1_1 = nn.PReLU()
        self.asy1_Conv1_3 = Conv1x3(inchannel // ratio, 1, 3)
        self.act1_2 = nn.PReLU()
        # generate spatial attention 2
        self.asy2_Conv1_3 = Conv1x3(inchannel, inchannel//ratio, 3)
        self.act2_1 = nn.PReLU()
        self.asy2_Conv3_1 = Conv3x1(inchannel // ratio, 1, 3)
        self.act2_2 = nn.PReLU()
        # generate spatial attention 3
        self.sy1_Conv3_3 = Conv3x3(inchannel, inchannel//ratio, 3)
        self.act3_1 = nn.PReLU()
        self.sy2_Conv3_3 = Conv3x3(inchannel // ratio, 1, 3)
        self.act3_2 = nn.PReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        spatial_att1 = self.act1_1(self.asy1_Conv3_1(x))
        spatial_att1 = self.act1_2(self.asy1_Conv1_3(spatial_att1))
        spatial_att2 = self.act2_1(self.asy2_Conv1_3(x))
        spatial_att2 = self.act2_2(self.asy2_Conv3_1(spatial_att2))
        spatial_att3 = self.act3_1(self.sy1_Conv3_3(x))
        spatial_att3 = self.act3_2(self.sy2_Conv3_3(spatial_att3))
        spatial_att = self.sig(spatial_att1 + spatial_att2 + spatial_att3)
        out = x*spatial_att  # broadcast

        return out + x


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, shortcut=None):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
                Conv3x3(inchannel, outchannel, 3),
                nn.PReLU(),
                Conv3x3(outchannel, outchannel, 3),
        )
        self.se = FirstSecondOrderMLP(outchannel, 16)
        self.right = shortcut
        self.act = nn.PReLU()

    def forward(self, x):
        out = self.left(x)
        out = self.se(out)
        residual = x if self.right is None else self.right(x)
        out += residual
        out = self.act(out)
        return out


class RAAUN(nn.Module):
    def __init__(self, in_features=3, out_features=31):
        super(RAAUN, self).__init__()
        self.in_conv = Conv3x3(in_features, 32, 3)

        self.encoder0 = self._makeLayer(32, 64, 2)
        self.encoder1 = self._makeLayer(64, 128, 2)
        self.encoder2 = self._makeLayer(128, 256, 2)
        self.encoder3 = self._makeLayer(256, 256, 2)
        self.encoder4 = self._makeLayer(256, 256, 2)

        self.encoder0_saa = SAA_AsySymConvResidual(64, 4)
        self.encoder1_saa = SAA_AsySymConvResidual(128, 4)
        self.encoder2_saa = SAA_AsySymConvResidual(256, 4)
        self.encoder3_saa = SAA_AsySymConvResidual(256, 4)
        self.encoder4_saa = SAA_AsySymConvResidual(256, 4)

        self.bottom = self._makeLayer(256, 256, 2)

        self.decoder0 = self._makeLayer(64, 32, 2)
        self.decoder1 = self._makeLayer(128, 64, 2)
        self.decoder2 = self._makeLayer(256, 128, 2)
        self.decoder3 = self._makeLayer(256, 256, 2)
        self.decoder4 = self._makeLayer(256, 256, 2)

        self.out_conv = Conv3x3(32, out_features, 3)

    def _encoder_path(self, inputs):
        lateral = []
        x = self.encoder0(inputs)
        lateral.append(x)
        x = self.encoder1(x)
        lateral.append(x)
        x = self.encoder2(x)
        lateral.append(x)
        x = self.encoder3(x)
        lateral.append(x)
        x = self.encoder4(x)
        lateral.append(x)
        return lateral

    def _decoder_path(self, inputs, lateral):
        x = self.decoder4(torch.add(inputs, self.encoder4_saa(lateral[4])))
        x = self.decoder3(torch.add(x, self.encoder3_saa(lateral[3])))
        x = self.decoder2(torch.add(x, self.encoder2_saa(lateral[2])))
        x = self.decoder1(torch.add(x, self.encoder1_saa(lateral[1])))
        x = self.decoder0(torch.add(x, self.encoder0_saa(lateral[0])))
        return x

    def _makeLayer(self, inchannel, outchannel, block_num):
        if inchannel == outchannel:
            shortcut = None
        else:
            shortcut = Conv3x3(inchannel, outchannel, 1)
        layers = []
        layers.append(ResBlock(inchannel, outchannel, shortcut))
        for i in range(1, block_num):
            layers.append(ResBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.in_conv(x)
        lateral = self._encoder_path(x)
        x = self.bottom(lateral[4])
        x = self._decoder_path(x, lateral)
        x = self.out_conv(x)
        return x


if __name__ == "__main__":
    # import os
    # os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    input_tensor = torch.rand(1, 3, 64, 64)
    model = RAAUN()
    # model = nn.DataParallel(model).cuda()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    print(output_tensor.size())
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    print(torch.__version__)


