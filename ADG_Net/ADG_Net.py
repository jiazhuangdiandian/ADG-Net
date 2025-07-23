import torch.nn.functional as F
import os
os.environ['CUDA_VISIBLE_DEVICE']='0'
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse
from torchvision.transforms.functional import rgb_to_grayscale
from DASCA import DASCA
from AIFM import AIFM
from MEEGA import MEEGA, make_laplace_pyramid


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):

        return self.conv(input)


class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True),
                                    )
    def forward(self, x):
        # print("000", x.shape)
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        # print("111", y_HH.shape)

        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        # print("222", x.shape)

        x = self.conv_bn_relu(x)
        # print("333", x.shape)

        return x


class Up_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up_wt, self).__init__()

        # DWT分解操作：分解为低频部分 yL 和 高频部分 yH
        self.wt = DWTForward(mode='zero', wave='haar')

        # 通过卷积、批量归一化和ReLU进行特征处理
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 执行DWT分解，得到低频部分 yL 和 高频部分 yH
        # print("Input x shape:", x.shape)
        yL, yH = self.wt(x)

        # 提取低频和高频部分
        y_L = yL
        y_HL = yH[0][:, :, 0, :]
        y_LH = yH[0][:, :, 1, :]
        y_HH = yH[0][:, :, 2, :]

        y_L_upsampled = F.interpolate(y_L, scale_factor=2, mode='bilinear', align_corners=False)

        y_HL_upsampled = F.interpolate(y_HL, scale_factor=2, mode='bilinear', align_corners=False)
        y_LH_upsampled = F.interpolate(y_LH, scale_factor=2, mode='bilinear', align_corners=False)
        y_HH_upsampled = F.interpolate(y_HH, scale_factor=2, mode='bilinear', align_corners=False)


        x_cat = torch.cat([y_L_upsampled, y_HL_upsampled, y_LH_upsampled, y_HH_upsampled], dim=1)


        x = self.conv_bn_relu(x_cat)

        return x



class Down_wt1(nn.Module):
    def __init__(self, in_ch):
        super(Down_wt1, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*4, 1, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(1),
                                    nn.ReLU(inplace=True),
                                    )
    def forward(self, x):
        # print("000", x.shape)
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        # print("111", x.shape)

        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        # print("222", x.shape)

        x = self.conv_bn_relu(x)
        # print("333", x.shape)

        return x


class Down_wt_laplace(nn.Module):
    def __init__(self, ):
        super(Down_wt_laplace, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(1),
                                    nn.ReLU(inplace=True),
                                    )

    def forward(self, x):


        grayscale_img = rgb_to_grayscale(x)
        edge_feature = make_laplace_pyramid(grayscale_img, 5, 1)

        edge_feature = edge_feature[1]
        out1 = edge_feature

        yL, yH = self.wt(edge_feature)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]

        edge_feature = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)

        x = self.conv_bn_relu(edge_feature)
        return x, out1

class Up_wt_laplace(nn.Module):
    def __init__(self, ):
        super(Up_wt_laplace, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(4, 1, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(1),
                                    nn.ReLU(inplace=True),
                                    )

    def forward(self, x):

        # print("000", x.shape)

        grayscale_img = rgb_to_grayscale(x)
        edge_feature = make_laplace_pyramid(grayscale_img, 5, 1)

        edge_feature = edge_feature[1]
        out1 = edge_feature

        yL, yH = self.wt(edge_feature)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]

        edge_feature = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)

        x = self.conv_bn_relu(edge_feature)

        return x, out1





class Detail_Guidance_Encoder(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            token_mixer=DASCA,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=AIFM,
            mlp_ratio=2,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,
            dw_size=11,
            square_kernel_size=3,
            ratio=1,
    ):
        super().__init__()
        self.token_mixer = token_mixer(dim, band_kernel_size=dw_size, square_kernel_size=square_kernel_size,
                                       ratio=ratio)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.Down_wt = Down_wt(dim, dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        shortcut = x
        shortcut = self.conv(x)
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)

        x = self.Down_wt(x)

        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x



class Detail_Guidance_Decoder(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            token_mixer=DASCA,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=AIFM,
            mlp_ratio=2,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,
            dw_size=11,
            square_kernel_size=3,
            ratio=1,
    ):
        super().__init__()
        self.token_mixer = token_mixer(dim, band_kernel_size=dw_size, square_kernel_size=square_kernel_size,
                                       ratio=ratio)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.Up_wt = Up_wt(dim, dim)


    def forward(self, x):
        shortcut = x

        # print("000", x.shape)
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        # print("111", x.shape)

        x = self.Up_wt(x)
        # print("222", x.shape)

        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


class DConv_Out(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # print("0", x.shape)
        x = self.conv(x)
        # print("1", x.shape)
        return x

class Out(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Out, self).__init__()
        self.conv1 = DConv_Out(in_channels, in_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


'''------------------------------------------ADG_Net-----------------------------------------------'''


class ADG_Net(nn.Module):


    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=384, patch_size=8, in_chans=3,
                 embed_dims=[32, 64, 128, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                     depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], test=True, height=384, **kwargs):
        super().__init__()
        size = [img_size, img_size//2, img_size//4, img_size//8, img_size//16]

        self.conv1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)

        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)

        self.RCM1 = Detail_Guidance_Encoder(32)
        self.RCM2 = Detail_Guidance_Encoder(64)
        self.RCM3 = Detail_Guidance_Encoder(128)
        self.RCM4 = Detail_Guidance_Encoder(256)

        self.RCM5 = Detail_Guidance_Decoder(256)
        self.RCM6 = Detail_Guidance_Decoder(128)
        self.RCM7 = Detail_Guidance_Decoder(64)
        self.RCM8 = Detail_Guidance_Decoder(32)

        self.ega1 = MEEGA(64)
        self.ega2 = MEEGA(128)
        self.ega3 = MEEGA(256)
        self.ega4 = MEEGA(512)

        self.out6 = Out(512, num_classes)
        self.out5 = Out(256, num_classes)
        self.out4 = Out(128, num_classes)
        self.out3 = Out(64, num_classes)



        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

        self.test = test

    def forward(self, x):
        grayscale_img = rgb_to_grayscale(x)
        edge_feature = make_laplace_pyramid(grayscale_img, 5, 1)
        edge_feature = edge_feature[1]

        c1 = self.conv1(x)
        c2 = self.conv2(self.RCM1(c1))  #  64

        c3 = self.conv3(self.RCM2(c2))   # 128


        c4 = self.conv4(self.RCM3(c3))  # 256


        c5 = self.conv5(self.RCM4(c4))   # 512

        out6 = self.out6(c5)
        ega4 = self.ega4(edge_feature, c5, out6)
        up_6 = self.up6(ega4)
        c6 = self.RCM5(up_6)

        out5 = self.out5(c6)
        ega3 = self.ega3(edge_feature, c4, out5)
        up_7 = self.up7(ega3)
        c7 = self.RCM6(up_7)

        out4 = self.out4(c7)
        ega2 = self.ega2(edge_feature, c3, out4)
        up_8 = self.up8(ega2)
        c8 = self.RCM7(up_8)

        out3 = self.out3(c8)
        ega1 = self.ega1(edge_feature, c2, out3)
        up_9 = self.up9(ega1)
        c9 = self.RCM8(up_9)

        out1 = self.final(c9)
        if self.test:
            return out1
        else:
            return out1, out3, out4, out5, out6








from ptflops import get_model_complexity_info
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ADG_Net(1).to(device)
flops, params = get_model_complexity_info(model, (3, 384, 384), as_strings=True, print_per_layer_stat=False)

print('flops: ', flops, 'params: ', params)


