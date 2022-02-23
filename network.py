import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import partial

from util.network_module import *

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


# ----------------------------------------
#                Generator
# ----------------------------------------
# Input enhancement block
class IEB(nn.Module):
    def __init__(self, opt, latent_dim = 1):
        super(IEB, self).__init__()

        # recover layer
        self.x_avgpooldown2_recover_layer = Conv2dLayer(opt.in_channels, opt.in_channels, 3, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.x_avgpooldown8_recover_layer = TransposeConv2dLayer(opt.in_channels, opt.in_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        
        # general layer
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
     
        # diff
        self.combine_layer = Conv2dLayer(int(opt.in_channels * (3 + 16)), opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.fc_diff_1 = nn.Linear(opt.start_channels, opt.start_channels // latent_dim)
        self.fc_diff_2 = nn.Linear(opt.start_channels // latent_dim, opt.start_channels)

        # x itself
        self.conv_x = Conv2dLayer(int(opt.in_channels * 16), opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
     
        # final concatenate
        self.final = Conv2dLayer(opt.start_channels * 2, opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)

    def forward(self, x_avgpooldown2, x_avgpooldown4, x_avgpooldown8, x_psdown4):
        
        # pre-processing
        x_avgpooldown2 = self.x_avgpooldown2_recover_layer(x_avgpooldown2)
        x_avgpooldown8 = self.x_avgpooldown8_recover_layer(x_avgpooldown8)

        # difference maps
        fea_comb_x = torch.cat((x_avgpooldown2, x_avgpooldown4, x_avgpooldown8, x_psdown4), 1)

        fea_comb_x = self.combine_layer(fea_comb_x)
        residual_fea_diff_x = fea_comb_x
        fea_comb_x = self.gap(fea_comb_x)
        fea_comb_x = fea_comb_x.view(fea_comb_x.size(0), -1)
        
        fea_comb_x = self.fc_diff_1(fea_comb_x)
        fea_comb_x = self.fc_diff_2(fea_comb_x)
        fea_comb_x = self.sigmoid(fea_comb_x)
        fea_comb_x = fea_comb_x.view(fea_comb_x.size(0), fea_comb_x.size(1), 1, 1)
        
        fea_comb_x = residual_fea_diff_x * fea_comb_x

        # x itself
        x_psdown4 = self.conv_x(x_psdown4)

        # final conv
        out = torch.cat((x_psdown4, fea_comb_x), 1)
        out = self.final(out)

        return out

class QRNet(nn.Module):
    def __init__(self, opt):
        super(QRNet, self).__init__()

        # PixelShuffle and its inverse
        self.pixel_shuffle_down = PixelUnShuffleAlign()
        self.pixel_shuffle_up = PixelShuffleAlign()

        # UNet
        self.short_conv4 = Conv2dLayer(opt.in_channels * 4, opt.start_channels // 4 * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.short_conv5 = Conv2dLayer(opt.in_channels, opt.start_channels // 4 * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)

        # IEB
        self.ieb_rgb = IEB(opt)

        # Encoder
        self.enc_rgb_1 = Conv2dLayer(opt.in_channels * 16, opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = 'none')
        self.enc_rgb_2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 3, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.enc_rgb_3_1 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 3, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.enc_rgb_3_2 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        
        # Bottleneck
        self.bottleneck_rgb_1_1 = nn.Sequential(
            ResConv2dLayer(opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        )
        self.bottleneck_rgb_1_2 = nn.Sequential(
            ResConv2dLayer(opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        )
        
        self.bottleneck_cat_2 = Conv2dLayer(opt.start_channels * (2 + 4), opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.bottleneck_rgb_2_1 = nn.Sequential(
            ResConv2dLayer(opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        )
        self.bottleneck_rgb_2_2 = nn.Sequential(
            ResConv2dLayer(opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        )
        
        self.bottleneck_cat_3 = Conv2dLayer(opt.start_channels * (1 + 2), opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.bottleneck_rgb_3_1 = ResConv2dLayer(opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.bottleneck_rgb_3_2 = ResConv2dLayer(opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        
        # Decoder rgb
        self.dec_rgb_1 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.dec_rgb_2 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.dec_rgb_3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.dec_rgb_4 = Conv2dLayer(opt.start_channels // 4 * 2 * 2, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.dec_rgb_5 = Conv2dLayer(opt.start_channels // 4 * 2 * 2, opt.out_channels, 3, 1, 1, pad_type = opt.pad, norm = 'none', activation = 'none')

    def forward(self, x):

        # input x shape: batch * 1 * 256 * 256
        short5_x = x

        # average pooling downsampling
        x_avgpooldown2 = F.interpolate(x, scale_factor = 0.5, mode = 'area')    # out: batch * 1 * 128 * 128
        x_avgpooldown4 = F.interpolate(x, scale_factor = 0.25, mode = 'area')   # out: batch * 1 * 64 * 64
        x_avgpooldown8 = F.interpolate(x, scale_factor = 0.125, mode = 'area')  # out: batch * 1 * 32 * 32

        # obtain 4 groups of RGGB(s), RGGB(l), RGGB(s), RGGB(l)
        # each group only contains one kind of exposure
        x_psdown2 = self.pixel_shuffle_down(x)                  # out: batch * 4 * 128 * 128
        short4_x = x_psdown2
        # obtain 16 groups of R(s), G(s), G(s), B(s), R(l), G(l), G(l), B(l), R(s), G(s), G(s), B(s), R(l), G(l), G(l), B(l)
        # each group only contains one kind of exposure, and one kind of color pixels
        x_psdown4 = self.pixel_shuffle_down(x_psdown2)          # out: batch * 16 * 64 * 64

        # enc1
        enc_rgb_1 = self.ieb_rgb(x_avgpooldown2, x_avgpooldown4, x_avgpooldown8, x_psdown4) # out: batch * 64 * 64 * 64

        # enc2
        enc_rgb_2 = self.enc_rgb_2(enc_rgb_1)                   # out: batch * 128 * 32 * 32

        # enc3
        enc_rgb_3 = self.enc_rgb_3_1(enc_rgb_2)                 # out: batch * 256 * 16 * 16
        enc_rgb_3 = self.enc_rgb_3_2(enc_rgb_3)                 # out: batch * 256 * 16 * 16

        # bottleneck
        bot1 = self.bottleneck_rgb_1_1(enc_rgb_3)               # out: batch * 256 * 16 * 16
        bot1_up = F.interpolate(bot1, scale_factor = 2, mode = 'bilinear')
        bot1 = self.bottleneck_rgb_1_2(bot1)                    # out: batch * 256 * 16 * 16

        # dec1
        bot1 = torch.cat((bot1, enc_rgb_3), 1)                  # out: batch * 512 * 16 * 16
        dec_rgb = self.dec_rgb_1(bot1)                          # out: batch * 128 * 32 * 32
        
        # dec2
        # bottleneck
        bot2 = torch.cat((enc_rgb_2, bot1_up), 1)               # out: batch * 384 * 32 * 32
        bot2 = self.bottleneck_cat_2(bot2)                      # out: batch * 128 * 32 * 32
        bot2 = self.bottleneck_rgb_2_1(bot2)                    # out: batch * 128 * 32 * 32
        bot2_up = F.interpolate(bot2, scale_factor = 2, mode = 'bilinear')
        bot2 = self.bottleneck_rgb_2_2(bot2)                    # out: batch * 128 * 32 * 32
        # concatenate and conv
        dec_rgb = torch.cat((dec_rgb, bot2), 1)                 # out: batch * 256 * 32 * 32
        dec_rgb = self.dec_rgb_2(dec_rgb)                       # out: batch * 64 * 64 * 64

        # dec3
        # bottleneck
        bot3 = torch.cat((enc_rgb_1, bot2_up), 1)               # out: batch * 192 * 64 * 64
        bot3 = self.bottleneck_cat_3(bot3)                      # out: batch * 64 * 64 * 64
        bot3 = self.bottleneck_rgb_3_1(bot3)                    # out: batch * 64 * 64 * 64
        bot3 = self.bottleneck_rgb_3_2(bot3)                    # out: batch * 64 * 64 * 64
        # concatenate and conv
        dec_rgb = torch.cat((dec_rgb, bot3), 1)                 # out: batch * 128 * 64 * 64
        dec_rgb = self.dec_rgb_3(dec_rgb)                       # out: batch * 128 * 64 * 64

        # dec4
        dec_rgb = self.pixel_shuffle_up(dec_rgb)                # out: batch * 32 * 128 * 128
        short4_x = self.short_conv4(short4_x)                   # out: batch * 32 * 128 * 128
        dec_rgb = torch.cat((dec_rgb, short4_x), 1)             # out: batch * 64 * 128 * 128
        dec_rgb = self.dec_rgb_4(dec_rgb)                       # out: batch * 128 * 128 * 128

        # dec5
        dec_rgb = self.pixel_shuffle_up(dec_rgb)                # out: batch * 32 * 256 * 256
        short5_x = self.short_conv5(short5_x)                   # out: batch * 32 * 256 * 256
        dec_rgb = torch.cat((dec_rgb, short5_x), 1)             # out: batch * 64 * 256 * 256
        dec_rgb = self.dec_rgb_5(dec_rgb)                       # out: batch * 3 * 256 * 256

        sample = {'dec_rgb': dec_rgb}

        return sample

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'zero', help = 'pad type of networks')
    parser.add_argument('--activ_g', type = str, default = 'lrelu', help = 'activation type of generator')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation type of discriminator')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 1, help = '1 for colorization, 3 for other tasks')
    parser.add_argument('--out_channels', type = int, default = 3, help = '2 for colorization, 3 for other tasks')
    parser.add_argument('--start_channels', type = int, default = 32, help = 'start channels for the main stream of generator')
    parser.add_argument('--init_type', type = str, default = 'normal', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    opt = parser.parse_args()
    
    net = QRNet(opt).cuda()
    x = torch.randn(1, 1, 320, 320).cuda()
    sample = net(x)
    top_out = sample['dec_rgb']
    print(top_out.shape)
