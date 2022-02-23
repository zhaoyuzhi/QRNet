import argparse
import torch
import yaml
from easydict import EasyDict as edict
from thop import profile
from thop import clever_format

import network

def attatch_to_config(opt, yaml_args):
    # Pre-train, saving, and loading parameters
    opt.network = yaml_args.name
    # Initialization parameters
    opt.pad = yaml_args.Network.pad
    opt.activ_g = yaml_args.Network.activ_g
    opt.activ_d = yaml_args.Network.activ_d
    opt.norm = yaml_args.Network.norm
    opt.in_channels = yaml_args.Network.in_channels
    opt.out_channels = yaml_args.Network.out_channels
    opt.start_channels = yaml_args.Network.start_channels
    opt.init_type = yaml_args.Network.init_type
    opt.init_gain = yaml_args.Network.init_gain

def create_generator_for_flops(opt):
    # Initialize the network
    generator = getattr(network, opt.network)(opt)
    network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
    return generator

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters (should be changed)
    parser.add_argument('--yaml_path', type = str, \
        default = './options/qrnetab3_raw1_data4000_loss4.yaml', \
            help = 'yaml_path')
    # Initialization parameters (just ignore all of them since parameters are recorded in option.yaml)
    parser.add_argument('--pad', type = str, default = 'zero', help = 'pad type of networks')
    parser.add_argument('--activ_g', type = str, default = 'lrelu', help = 'activation type of generator')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation type of discriminator')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 1, help = '1 for quadbayer, 3 for rgb')
    parser.add_argument('--out_channels', type = int, default = 3, help = '1 for quadbayer, 3 for rgb')
    parser.add_argument('--start_channels', type = int, default = 16, help = 'start channels for the main stream of generator')
    parser.add_argument('--init_type', type = str, default = 'normal', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    # Flops parameters (should be changed)
    parser.add_argument('--h', type = int, default = 512, help = 'h resolution')
    parser.add_argument('--w', type = int, default = 512, help = 'w resolution')
    opt = parser.parse_args()
    
    # name: ./options/qrnet_raw1_data4000_loss4.yaml macs: 34.628G params: 13.363M
    # name: ./options/qrnetab3_raw1_data4000_loss4.yaml macs: 33.084G
    # name: ./options/qrnetab4_raw1_data4000_loss4.yaml macs: 31.944G
    # name: ./options/qrnetab5_raw1_data4000_loss4.yaml macs: 21.539G
    # name: ./options/qrnetab6_raw1_data4000_loss4.yaml macs: 12.475G

    with open(opt.yaml_path, mode = 'r') as f:
        yaml_args = edict(yaml.load(f))

    attatch_to_config(opt, yaml_args)

    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    
    # Define the network
    generator = create_generator_for_flops(opt).cuda()

    for param in generator.parameters():
        param.requires_grad = False

    # forward propagation
    input = torch.randn(1, 1, opt.h, opt.w).cuda()

    macs, params = profile(generator, inputs = (input, ))
    macs_1, params_1 = clever_format([macs, params], "%.3f")
    print('name:', opt.yaml_path, 'macs:', macs_1, 'params:', params_1)
