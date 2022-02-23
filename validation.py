import argparse
import yaml
from easydict import EasyDict as edict
import os

import trainer
import utils

def get_load_name(yaml_args):
    # Data settings
    num_train_img, num_blur_patch, num_salient_patch, num_joint_patch \
        = yaml_args.Dataset.num_input[0], yaml_args.Dataset.num_input[1], yaml_args.Dataset.num_input[2], yaml_args.Dataset.num_input[3]
    str_num = str(num_train_img) + str(num_blur_patch) + str(num_salient_patch) + str(num_joint_patch)
    # Loss settings
    if yaml_args.Training.loss_grad_list[0] > 0 or yaml_args.Training.loss_grad_list[1] > 0:
        str_loss = '4'
    else:
        str_loss = '1'
    # configurations
    folder_name = '%s__K%0.2f__sigma%0.1f__data%s__loss%s' % \
        (yaml_args.name, yaml_args.Dataset.noise_K, yaml_args.Dataset.noise_sigma, str_num, str_loss)
    #folder_name = 'QRNet_ab7__K0.25__sigma5.0__data4000__loss1'
    return folder_name

def get_model_name(yaml_args):
    model_name_keyword = '%s_epoch%d_bs' % (yaml_args.name, yaml_args.Training.epochs) #yaml_args.Training.epochs
    folder_name = get_load_name(yaml_args)
    model_folder = os.path.join(yaml_args.load_name, folder_name)
    print(model_folder)
    model_name_list = utils.get_files(model_folder)
    for i in range(len(model_name_list)):
        print(model_name_list[i])
        if model_name_keyword in model_name_list[i]:
            return model_name_list[i]

def attatch_to_config(opt, yaml_args):
    # Pre-train, saving, and loading parameters
    opt.network = yaml_args.name
    opt.load_name = get_model_name(yaml_args)
    # Validation parameters
    opt.whether_save = yaml_args.Validation.whether_save
    opt.saveroot = yaml_args.Validation.saveroot
    opt.val_batch_size = 1
    opt.num_workers = 0
    opt.enable_patch = yaml_args.Validation.enable_patch
    opt.patch_size = yaml_args.Validation.patch_size
    # Initialization parameters
    opt.pad = yaml_args.Network.pad
    opt.activ_g = yaml_args.Network.activ_g
    opt.activ_d = yaml_args.Network.activ_d
    opt.norm = yaml_args.Network.norm
    opt.in_channels = yaml_args.Network.in_channels
    opt.out_channels = yaml_args.Network.out_channels
    opt.start_channels = yaml_args.Network.start_channels
    # Dataset parameters
    opt.baseroot_val = yaml_args.Dataset.baseroot_val
    opt.short_expo_per_pattern = yaml_args.Dataset.short_expo_per_pattern
    opt.noise_K = yaml_args.Dataset.noise_K
    opt.noise_sigma = yaml_args.Dataset.noise_sigma
    if hasattr(yaml_args.Dataset, 'ablation_data'):
        opt.ablation_data = yaml_args.Dataset.ablation_data
    else:
        opt.ablation_data = '_quadbayer_ls'

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--yaml_path', type = str, \
        default = './options/qrnet_raw1_data1111_loss11.yaml', \
            help = 'yaml_path')
    parser.add_argument('--network', type = str, default = 'QRNet', help = 'network name')
    parser.add_argument('--load_name', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    # Validation parameters
    parser.add_argument('--whether_save', type = bool, default = True, help = 'whether saving generated images')
    parser.add_argument('--saveroot', type = str, default = './val_results', help = 'saving path that is a folder')
    parser.add_argument('--val_batch_size', type = int, default = 1, help = 'val_batch_size, fixed to 1')
    parser.add_argument('--num_workers', type = int, default = 0, help = 'num_workers, fixed to 0')
    parser.add_argument('--enable_patch', type = bool, default = True, help = 'whether use patch for validation')
    parser.add_argument('--patch_size', type = int, default = 512, help = 'patch_size')
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'zero', help = 'pad type of networks')
    parser.add_argument('--activ_g', type = str, default = 'relu', help = 'activation type of generator')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation type of discriminator')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = '1 for quadbayer, 3 for rgb')
    parser.add_argument('--out_channels', type = int, default = 3, help = '1 for quadbayer, 3 for rgb')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    # Dataset parameters
    parser.add_argument('--baseroot_val', type = str, \
        default = 'E:\\submitted papers\\QuadBayer Deblur\\data\\val', \
            help = 'output image baseroot')
    parser.add_argument('--short_expo_per_pattern', type = int, default = 2, help = 'the number of exposure pixel of 2*2 square')
    parser.add_argument('--noise_K', type = float, default = 0.25, help = 'the noise parameter K')
    parser.add_argument('--noise_sigma', type = float, default = 5, help = 'the noise parameter sigma')
    parser.add_argument('--ablation_data', type = str, default = '', help = 'ablation_data')
    opt = parser.parse_args()

    with open(opt.yaml_path, mode = 'r') as f:
        yaml_args = edict(yaml.load(f))

    attatch_to_config(opt, yaml_args)
    print(opt)

    trainer.Valer(opt)
    