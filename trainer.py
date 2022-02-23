import time
import datetime
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import dataset
import utils
from util.network_module import *
from util.loss import *

def Trainer(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------
    
    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Data settings
    num_train_img, num_blur_patch, num_salient_patch, num_joint_patch \
        = opt.num_input[0], opt.num_input[1], opt.num_input[2], opt.num_input[3]
    str_num = str(num_train_img) + str(num_blur_patch) + str(num_salient_patch) + str(num_joint_patch)

    # Loss settings
    if opt.loss_grad_list[0] > 0 or opt.loss_grad_list[1] > 0:
        str_loss = '4'
    else:
        str_loss = '1'
    
    # Ablation study settings
    if 'qrnetab' in opt.yaml_path and opt.network == 'QRNet':
        str_ab = '_ab' + opt.yaml_path.split('qrnetab')[-1][0]
    else:
        str_ab = ''

    # configurations
    tail_name = '%s__K%0.2f__sigma%0.1f__data%s__loss%s' % \
        (opt.network + str_ab, opt.noise_K, opt.noise_sigma, str_num, str_loss)
    save_folder = os.path.join(opt.save_path, tail_name)
    sample_folder = os.path.join(opt.sample_path, tail_name)
    utils.check_path(save_folder)
    utils.check_path(sample_folder)

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()
    criterion_FFT = FFTLoss().cuda()

    # Initialize Generator
    generator = utils.create_generator(opt)
    pixel_shuffle_down = PixelUnShuffleAlign(2)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
        pixel_shuffle_down = nn.DataParallel(pixel_shuffle_down)
        pixel_shuffle_down = pixel_shuffle_down.cuda()
    else:
        generator = generator.cuda()
        pixel_shuffle_down = pixel_shuffle_down.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, optimizer):
        target_epoch = opt.epochs - opt.lr_decrease_epoch
        remain_epoch = opt.epochs - epoch
        if epoch >= opt.lr_decrease_epoch:
            lr = opt.lr_g * remain_epoch / target_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        # Define the name of trained model
        if opt.save_mode == 'epoch':
            model_name = '%s_epoch%d_bs%d.pth' % (opt.network, epoch, opt.train_batch_size * 4)
        if opt.save_mode == 'iter':
            model_name = '%s_iter%d_bs%d.pth' % (opt.network, iteration, opt.train_batch_size * 4)
        save_model_path = os.path.join(save_folder, model_name)
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.train_batch_size = opt.train_batch_size * gpu_num
    opt.num_workers = opt.num_workers * gpu_num

    # Define the dataset
    trainset = dataset.QuadBayer2RGB_Dataset(opt)
    print('The overall number of training images:', len(trainset))

    # Define the dataloader
    train_loader = DataLoader(trainset, batch_size = opt.train_batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    
    # For loop training
    for epoch in range(opt.epochs):
        for i, data in enumerate(train_loader):

            # To device
            input_batch = data['input_batch'].cuda()
            rgbout_batch = data['rgbout_batch'].cuda()

            # Process patch
            if len(input_batch.shape) == 5:
                _, _, _, H, W = input_batch.shape # B, B, C, H, W
                input_batch = input_batch.view(-1, 1, H, W)
                rgbout_batch = rgbout_batch.view(-1, 3, H, W)
            
            # Train Generator
            optimizer_G.zero_grad()
            sample = generator(input_batch)
            rgbout_gen = sample['dec_rgb']

            # Compute losses
            rgb_loss = opt.loss_list[0] * criterion_L1(rgbout_gen, rgbout_batch)
            
            if opt.loss_grad_list[0] > 0 or opt.loss_grad_list[1] > 0:
                fft_loss = (opt.loss_list[0] * opt.loss_grad_list[1]) * criterion_FFT(rgbout_gen, rgbout_batch)
                loss = rgb_loss + fft_loss
            else:
                loss = rgb_loss
            
            # Overall Loss and optimize
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(train_loader) + i
            iters_left = opt.epochs * len(train_loader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            if opt.loss_grad_list[0] > 0 or opt.loss_grad_list[1] > 0:
                print("\r[Epoch %d/%d] [Batch %d/%d] [Full Loss: %.4f] [RGB Loss: %.4f] [FFT Loss: %.4f] Time_left: %s" %
                    ((epoch + 1), opt.epochs, i, len(train_loader), loss.item(), rgb_loss.item(), fft_loss.item(), time_left))
            else:
                print("\r[Epoch %d/%d] [Batch %d/%d] [Full Loss: %.4f] [RGB Loss: %.4f] Time_left: %s" %
                    ((epoch + 1), opt.epochs, i, len(train_loader), loss.item(), rgb_loss.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(train_loader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), optimizer_G)
            
        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [input_batch, rgbout_gen, rgbout_batch]
            name_list = ['input', 'pred', 'gt']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'train_epoch%d' % (epoch + 1), \
                img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

def Valer(opt):

    # ----------------------------------------
    #                Prepararing
    # ----------------------------------------
    
    # configurations
    tail_name = opt.load_name.split('/')[-2] # e.g., QRNet__K0.25__sigma5.0__data1111__loss11
    save_folder = os.path.join(opt.saveroot, tail_name)
    if opt.whether_save:
        utils.check_path(save_folder)

    # Initialize Generator
    generator = utils.create_generator(opt)

    # To device
    generator = generator.cuda()

    # Define the dataset
    valset = dataset.QuadBayer2RGB_Valset(opt)
    print('The overall number of validation images:', len(valset))

    # Define the dataloader
    val_loader = DataLoader(valset, batch_size = opt.val_batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                Validation
    # ----------------------------------------

    # forward
    val_PSNR = 0
    val_SSIM = 0
    
    # For loop training
    for i, (true_input, true_target, save_img_path) in enumerate(val_loader):

        # To device
        true_input = true_input.cuda()
        true_target = true_target.cuda()
        save_img_path = save_img_path[0]
        
        # Forward
        with torch.no_grad():
            if opt.enable_patch:
                _, _, H, W = true_input.shape 
                patch_size = opt.patch_size
                assert patch_size % 4 == 0
                patchGen = utils.PatchGenerator(H, W, patch_size)
                out = torch.zeros_like(true_target)
                for (h, w, top_padding, left_padding, bottom_padding, right_padding) in patchGen.next_patch():
                    img_patch = true_input[:, :, h:h+patch_size, w:w+patch_size]
                    sample = generator(img_patch)
                    out_patch = sample['dec_rgb']
                    out[:, :, h+top_padding:h+patch_size-bottom_padding, w+left_padding:w+patch_size-right_padding] = \
                        out_patch[:, :, top_padding:patch_size-bottom_padding, left_padding:patch_size-right_padding]
            else:
                sample = generator(true_input)
                out = sample['dec_rgb']
        
        # Save the image (BCHW -> HWC)
        if opt.whether_save:
            save_img = out[0, :, :, :].clone().data.permute(1, 2, 0).cpu().numpy()
            save_img = np.clip(save_img, 0, 1)
            save_img = (save_img * 255).astype(np.uint8)
            save_full_path = os.path.join(save_folder, save_img_path)
            cv2.imwrite(save_full_path, save_img)

        # PSNR
        # print('The %d-th image PSNR %.4f' % (i, val_PSNR_this))
        this_PSNR = utils.psnr(out, true_target, 1) * true_target.shape[0]
        val_PSNR += this_PSNR
        this_SSIM = utils.ssim(out, true_target) * true_target.shape[0]
        val_SSIM += this_SSIM
        print('The %d-th image: Name: %s PSNR: %.5f, SSIM: %.5f' % (i + 1, save_img_path, this_PSNR, this_SSIM))

    val_PSNR = val_PSNR / len(valset)
    val_SSIM = val_SSIM / len(valset)
    print('The average of %s: PSNR: %.5f, average SSIM: %.5f' % (opt.load_name, val_PSNR, val_SSIM))
