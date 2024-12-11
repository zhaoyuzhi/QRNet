import argparse
import yaml
from easydict import EasyDict as edict
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import trainer

def attatch_to_config(opt, yaml_args):
    # Pre-train, saving, and loading parameters
    opt.network = yaml_args.name
    opt.save_path = yaml_args.Training.save_path
    opt.sample_path = yaml_args.Training.sample_path
    opt.save_mode = yaml_args.Training.save_mode
    opt.save_by_epoch = yaml_args.Training.save_by_epoch
    opt.save_by_iter = yaml_args.Training.save_by_iter
    if yaml_args.Training.stage == 1:
        opt.load_name = ""
    elif yaml_args.Training.stage == 2:
        opt.network_first_stage_net = yaml_args.name_first_stage_net
        opt.load_name_first_stage_net = yaml_args.load_name_first_stage_net
    opt.multi_gpu = yaml_args.Training.multi_gpu
    opt.cudnn_benchmark = yaml_args.Training.cudnn_benchmark
    # Training parameters
    opt.stage = yaml_args.Training.stage
    opt.epochs = yaml_args.Training.epochs
    opt.train_batch_size = yaml_args.Training.train_batch_size
    opt.val_batch_size = yaml_args.Training.val_batch_size
    opt.lr_g = yaml_args.Training.lr_g
    opt.lr_d = yaml_args.Training.lr_d
    opt.b1 = yaml_args.Training.b1
    opt.b2 = yaml_args.Training.b2
    opt.weight_decay = yaml_args.Training.weight_decay
    opt.lr_decrease_epoch = yaml_args.Training.lr_decrease_epoch
    opt.num_workers = yaml_args.Training.num_workers
    opt.loss_list = yaml_args.Training.loss_list
    opt.loss_grad_list = yaml_args.Training.loss_grad_list
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
    # Dataset parameters
    opt.baseroot_train = yaml_args.Dataset.baseroot_train
    opt.baseroot_train_blur_patch = yaml_args.Dataset.baseroot_train_blur_patch
    opt.baseroot_train_salient_patch = yaml_args.Dataset.baseroot_train_salient_patch
    opt.baseroot_train_joint_patch = yaml_args.Dataset.baseroot_train_joint_patch
    opt.baseroot_val = yaml_args.Dataset.baseroot_val
    opt.crop_size = yaml_args.Training.crop_size
    opt.num_input = yaml_args.Dataset.num_input
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
    parser.add_argument('--yaml_path', type = str, default = './options/qrnet_raw1_data4000_loss4.yaml', help = 'yaml_path')
    parser.add_argument('--network', type = str, default = 'QRNet', help = 'network name')
    parser.add_argument('--save_path', type = str, default = './models', help = 'saving path that is a folder')
    parser.add_argument('--sample_path', type = str, default = './samples', help = 'training samples path that is a folder')
    parser.add_argument('--save_mode', type = str, default = 'epoch', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 10, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--save_by_iter', type = int, default = 100000, help = 'interval between model checkpoints (by iterations)')
    parser.add_argument('--load_name', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    parser.add_argument('--network_first_stage_net', type = str, default = '', help = 'network name')
    parser.add_argument('--load_name_first_stage_net', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    # GPU parameters
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'True for more than 1 GPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    # Training parameters
    parser.add_argument('--stage', type = int, default = 1, help = 'training stage 1 / 2')
    parser.add_argument('--epochs', type = int, default = 300, help = 'number of epochs of training')
    parser.add_argument('--train_batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--val_batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--lr_g', type = float, default = 0.0001, help = 'Adam: learning rate for G / D')
    parser.add_argument('--lr_d', type = float, default = 0.0001, help = 'Adam: learning rate for G / D')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay for optimizer')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 150, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--num_workers', type = int, default = 0, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--loss_list', type = list, default = [1, 1], help = 'coefficient for Loss for different branches')
    parser.add_argument('--loss_grad_list', type = list, default = [0.1, 0.01], help = 'coefficient for Loss for different branches')
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'zero', help = 'pad type of networks')
    parser.add_argument('--activ_g', type = str, default = 'relu', help = 'activation type of generator')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation type of discriminator')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = '1 for quadbayer, 3 for rgb')
    parser.add_argument('--out_channels', type = int, default = 3, help = '1 for quadbayer, 3 for rgb')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    parser.add_argument('--init_type', type = str, default = 'normal', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    # Dataset parameters
    parser.add_argument('--baseroot_train', type = str, default = './QRNet-release/train', help = 'input image baseroot')
    parser.add_argument('--baseroot_train_blur_patch', type = str, default = './QRNet-release/train_blur_patch', help = 'input image baseroot')
    parser.add_argument('--baseroot_train_salient_patch', type = str, default = './QRNet-release/train_salient_patch', help = 'input image baseroot')
    parser.add_argument('--baseroot_train_joint_patch', type = str, default = './QRNet-release/train_joint_patch', help = 'input image baseroot')
    parser.add_argument('--baseroot_val', type = str, default = './QRNet-release/noisy_input', help = 'output image baseroot')
    parser.add_argument('--crop_size', type = int, default = 320, help = 'crop_size')
    parser.add_argument('--num_input', type = list, default = [4, 0, 0, 0], help = 'number of different input')
    parser.add_argument('--short_expo_per_pattern', type = int, default = 2, help = 'the number of exposure pixel of 2*2 square')
    parser.add_argument('--noise_K', type = float, default = 0.25, help = 'the noise parameter K')
    parser.add_argument('--noise_sigma', type = float, default = 5, help = 'the noise parameter sigma')
    parser.add_argument('--ablation_data', type = str, default = '', help = 'ablation_data')
    opt = parser.parse_args()

    with open(opt.yaml_path, mode = 'r') as f:
        yaml_args = edict(yaml.load(f, Loader = yaml.FullLoader))

    attatch_to_config(opt, yaml_args)
    print(opt)

    trainer.Trainer(opt)
    