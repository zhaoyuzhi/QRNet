import os
import cv2
import skimage.measure
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv

import network

# ----------------------------------------
#                 Network
# ----------------------------------------
def create_generator(opt):
    # Initialize the network
    generator = getattr(network, opt.network)(opt)
    if opt.load_name:
        # Load a pre-trained network
        pretrained_net = torch.load(opt.load_name)
        load_dict(generator, pretrained_net)
        print('Generator is loaded!')
    else:
        # Init the network
        network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Generator is created!')
    return generator

def create_generator_first_stage_net(opt): # only used for second training stage
    # Initialize the network
    generator = getattr(network, opt.network_first_stage_net)(opt)
    # Load a pre-trained network
    pretrained_net = torch.load(opt.load_name_first_stage_net)
    load_dict(generator, pretrained_net)
    print('Generator is loaded!')
    # It does not need weights
    for param in generator.parameters():
        param.requires_grad = False
    return generator

def create_discriminator(opt):
    # Initialize the network
    discriminator = network.PatchDiscriminator70(opt)
    # Init the network
    network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('Discriminators is created!')
    return discriminator
    
def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net

# ----------------------------------------
#    Validation and Sample at training
# ----------------------------------------
def save_sample_png(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization
        img = img * 255.0
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)[0, :, :, :]
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)

def save_sample_png_test(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization
        img = img * 255.0
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)[0, :, :, :]
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)

def psnr(pred, target, pixel_max_cnt = 255):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p

def grey_psnr(pred, target, pixel_max_cnt = 255):
    pred = torch.sum(pred, dim = 0)
    target = torch.sum(target, dim = 0)
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt * 3 / rmse_avg)
    return p

def ssim(pred, target):
    pred = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target[0]
    pred = pred[0]
    ssim = skimage.measure.compare_ssim(target, pred, multichannel = True)
    return ssim

# ----------------------------------------
#             PATH processing
# ----------------------------------------
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = [] 
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret
    
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

# ----------------------------------------
#              PatchGenerator
# ----------------------------------------
class PatchGenerator(object):
    def __init__(self, H, W, patch_size=None, padding=16):
        # assert H == W and H % 2 == 0
        self.H = H
        self.W = W
        self.padding = padding
        self.patch_size = self._calc_patch_size(patch_size)
    def _calc_patch_size(self, patch_size):
        if patch_size is None:
            assert self.padding is not None
            patch_size = self.H // 2 + self.padding
            return patch_size  
        else:
            return patch_size
    def next_patch(self):
        H, W = self.H, self.W
        padding = self.padding
        patch_size = self.patch_size
        H_block_num = int(np.ceil((H - padding * 2) / (patch_size - padding * 2)))
        W_block_num = int(np.ceil((W - padding * 2) / (patch_size - padding * 2)))
        for i in range(H_block_num):
            h = i * (patch_size - 2 * padding)
            if i == 0:
                h = 0
            elif i == H_block_num - 1:
                h = H - patch_size
            top_padding, bottom_padding = padding, padding
            if i == 0:
                top_padding = 0
            elif i == H_block_num - 1:
                bottom_padding = 0
            for j in range(W_block_num):
                w = j * (patch_size - 2 * padding)
                if j == 0:
                    w = 0
                elif j == W_block_num - 1:
                    w = W - patch_size
                left_padding, right_padding = padding, padding
                if j == 0:
                    left_padding = 0
                elif j == W_block_num - 1:
                    right_padding = 0
                yield h, w, top_padding, left_padding, bottom_padding, right_padding

def PatchGenerator_QuadBayer(H, W, patch_size):
    h_list = []
    w_list = []
    H_block_num = int(np.ceil((H) / (patch_size)))
    W_block_num = int(np.ceil((W) / (patch_size)))
    #out = np.zeros((H, W), dtype = np.uint8)
    #k = 0
    for i in range(H_block_num):
        h = patch_size * i
        if h + patch_size > H:
            h = H - patch_size
        for j in range(W_block_num):
            w = patch_size * j
            if w + patch_size > W:
                w = W - patch_size
            h_list.append(h)
            w_list.append(w)
            #print(h, w)
            #out[h:h+patch_size, w:w+patch_size] = 129
            #cv2.imwrite(str(k) + '.png', out)
            #k = k + 1
    return h_list, w_list

if __name__ == "__main__":
    
    H, W = 2010, 3018
    patch_size = 512
    assert patch_size % 4 == 0
    '''
    patchGen = PatchGenerator(H, W, patch_size)
    for (h, w, top_padding, left_padding, bottom_padding, right_padding) in patchGen.next_patch():
        print(h, w, top_padding, left_padding, bottom_padding, right_padding)
    '''
    h_list, w_list = PatchGenerator_QuadBayer(H, W, patch_size)
    print(h_list)
    print(w_list)
