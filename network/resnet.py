import torch
import torch.nn as nn
import torch.nn.init as init
import math
import numpy as np
from network.modules import *

class ResBlock(nn.Module):

    def __init__(self, nin, nout):
        super().__init__()

        self.bn =  nn.Sequential(
            #nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(nin, nout, 3, padding = 1), #conv1 in bottleneck
            nn.BatchNorm2d(nout),
            nn.ReLU(inplace=True),
            nn.Conv2d(nout, nout, 3, padding = 1),
            nn.BatchNorm2d(nout),
        )

        self.residual = nn.Sequential()

        if nin != nout:
            self.residual = nn.Sequential(
                nn.Conv2d(nin,nout,kernel_size=1,bias=False),
                nn.BatchNorm2d(nout),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.bn(x)+self.residual(x))

class Deconv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv =  nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding = 1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.deconv(x)

class ResEncoder(nn.Module):
    def __init__(self, num_channels, nc):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(num_channels, nc, 3, padding=1),
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc, nc, 3, padding=1),
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True),
        )

        self.second_1 = ResBlock(nc, nc*2)
        self.second_2 = ResBlock(nc*2, nc*2)

        self.third_1 = ResBlock(nc*2, nc*4)
        self.third_2 = ResBlock(nc*4, nc*4)

        self.fourth_1 = ResBlock(nc*4, nc*8)
        self.fourth_2 = ResBlock(nc*8, nc*8)

        self.fifth_1 = ResBlock(nc*8,nc*8)
        self.fifth_2 = ResBlock(nc*8,nc*8)

        self.sixth_1 = ResBlock(nc*8,nc*8)
        self.sixth_2 = ResBlock(nc*8,nc*8)

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)


    def forward(self, x):
        scale1 = self.first(x)
        scale2 = self.second_2(self.second_1(self.pool(scale1)))
        scale3 = self.third_2(self.third_1(self.pool(scale2)))
        scale4 = self.fourth_2(self.fourth_1(self.pool(scale3)))
        scale5 = self.fifth_2(self.fifth_1(self.pool(scale4)))
        scale6 = self.sixth_2(self.sixth_1(self.pool(scale5)))

        return scale1,scale2,scale3,scale4,scale5,scale6

class FANetHead(nn.Module):  # fusion with phase attention
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(FANetHead, self).__init__()
        inter_channels = in_channels // 4

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv5p = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sc = CAM_Module(inter_channels)  # intra-phase attention
        self.pa = PA_Module(inter_channels)  # inter-phase attention
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

    def forward(self, x_pv, x_art):
        feat1 = self.conv5c(x_pv)
        sc_feat = self.sc(feat1)
        sc_conv = self.conv51(sc_feat)

        feat2 = self.conv5p(x_art)
        pa_feat = self.pa(feat1,feat2)
        pa_conv = self.conv52(pa_feat)

        output = torch.cat([sc_conv, pa_conv], 1)

        return output

class RESNET_2D(nn.Module):  # 2D-Resnet

    def __init__(self, num_classes, num_channels, nc):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(num_channels, nc, 3, padding=1),
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc, nc, 3, padding=1),
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True),
        )

        self.second_1 = ResBlock(nc, nc*2)
        self.second_2 = ResBlock(nc*2, nc*2)

        self.third_1 = ResBlock(nc*2, nc*4)
        self.third_2 = ResBlock(nc*4, nc*4)

        self.fourth_1 = ResBlock(nc*4, nc*8)
        self.fourth_2 = ResBlock(nc*8, nc*8)

        self.fifth_1 = ResBlock(nc*8,nc*8)
        self.fifth_2 = ResBlock(nc*8,nc*8)

        self.sixth_1 = ResBlock(nc*8,nc*8)
        self.sixth_2 = ResBlock(nc*8,nc*8)

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.up1 = nn.Sequential(
            nn.Conv2d(nc, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.up2 = self.up_layer(nin=nc*2,nout=16,num_up=1)
        self.up3 = self.up_layer(nin=nc*4,nout=16,num_up=2)
        self.up4 = self.up_layer(nin=nc*8,nout=16,num_up=3)
        self.up5 = self.up_layer(nin=nc*8,nout=16,num_up=4)
        self.up6 = self.up_layer(nin=nc*8,nout=16,num_up=5)

        self.final = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,num_classes,kernel_size=1,padding=0),
        )

        #self._initialize_weights()

    def up_layer(self,nin, nout, num_up):
        cur_in = nin
        cur_out = ((2 ** (num_up - 1)) * nout)

        layers = []
        for i in range(num_up):
            layers.append(Deconv(cur_in, cur_out))
            cur_in = cur_out
            cur_out = math.floor(cur_out / 2)
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                # print(m.bias)
                # if m.bias:
                init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        scale1 = self.first(x)
        scale2 = self.second_2(self.second_1(self.pool(scale1)))
        scale3 = self.third_2(self.third_1(self.pool(scale2)))
        scale4 = self.fourth_2(self.fourth_1(self.pool(scale3)))
        scale5 = self.fifth_2(self.fifth_1(self.pool(scale4)))
        scale6 = self.sixth_2(self.sixth_1(self.pool(scale5)))

        up1 = self.up1(scale1)
        up2 = self.up2(scale2)
        up3 = self.up3(scale3)
        up4 = self.up4(scale4)
        up5 = self.up5(scale5)
        up6 = self.up6(scale6)

        concat_up = torch.cat([up1,up2,up3,up4,up5,up6],1)

        return self.final(concat_up)


class RESNET_phase_att(nn.Module):  # phase attention 20200414
    def __init__(self, num_classes, num_channels, nc):
        super().__init__()
        self.PV_encoder = ResEncoder(num_channels, nc)
        self.ART_encoder = ResEncoder(num_channels, nc)


        self.cf1 = FANetHead(nc)
        self.cf2 = FANetHead(nc*2)
        self.cf3 = FANetHead(nc*4)
        self.cf4 = FANetHead(nc*8)
        self.cf5 = FANetHead(nc*8)
        self.cf6 = FANetHead(nc*8)

        self.up1 = nn.Sequential(
            nn.Conv2d(int(nc * 2/4), 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.up2 = self.up_layer(nin=int(nc * 2 * 2/4), nout=16, num_up=1)
        self.up3 = self.up_layer(nin=int(nc * 4 * 2/4), nout=16, num_up=2)
        self.up4 = self.up_layer(nin=int(nc * 8 * 2/4), nout=16, num_up=3)
        self.up5 = self.up_layer(nin=int(nc * 8 * 2/4), nout=16, num_up=4)
        self.up6 = self.up_layer(nin=int(nc * 8 * 2/4), nout=16, num_up=5)


        self.final = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, num_classes, kernel_size=1, padding=0),
        )
        # self._initialize_weights()

    def up_layer(self, nin, nout, num_up):
        cur_in = nin
        cur_out = ((2 ** (num_up - 1)) * nout)

        layers = []
        for i in range(num_up):
            layers.append(Deconv(cur_in, cur_out))
            cur_in = cur_out
            cur_out = math.floor(cur_out / 2)
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                # print(m.bias)
                # if m.bias:
                init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x_pv, x_art):
        scale1, scale2, scale3, scale4, scale5, scale6 = self.PV_encoder(x_pv)
        c1, c2, c3, c4, c5, c6 = self.ART_encoder(x_art)

        f1 = self.cf1(scale1, c1)
        f2 = self.cf2(scale2, c2)
        f3 = self.cf3(scale3, c3)
        f4 = self.cf4(scale4, c4)
        f5 = self.cf5(scale5, c5)
        f6 = self.cf6(scale6, c6)

        up1 = self.up1(f1)
        up2 = self.up2(f2)
        up3 = self.up3(f3)
        up4 = self.up4(f4)
        up5 = self.up5(f5)
        up6 = self.up6(f6)

        concat_up = torch.cat([up1, up2, up3, up4, up5, up6], 1)

        return self.final(concat_up)


