import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset

import util.dataset_aug as DA

def get_heads_train(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            head = filespath.split('_')[0]
            if head not in ret:
                ret.append(head)
    return ret

def get_heads_train_patch(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            head = filespath.split('_')[0] + '_' + filespath.split('_')[1]
            if head not in ret:
                ret.append(head)
    return ret

def get_raw_test(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if filespath[-4:] == '.raw':
                ret.append(os.path.join(root, filespath))
    return ret

class QuadBayer2RGB_Dataset(Dataset):
    def __init__(self, opt):
        self.opt = opt

        # Build training dataset
        self.train_list = get_heads_train(opt.baseroot_train)
        self.train_blur_patch_list = get_heads_train_patch(opt.baseroot_train_blur_patch)
        self.train_salient_patch_list = get_heads_train_patch(opt.baseroot_train_salient_patch)
        self.train_joint_patch_list = get_heads_train_patch(opt.baseroot_train_joint_patch)

        # Specify the pos for short and long exposure pixels
        if opt.short_expo_per_pattern == 2:
            self.short_pos = [[0,0], [1,1]]
            self.long_pos = [[0,1], [1,0]]
        if opt.short_expo_per_pattern == 3:
            self.short_pos = [[0,0], [0,1], [1,0]]
            self.long_pos = [[1,1]]

    def random_crop_start(self, h, w, crop_size, min_divide):
        rand_h = random.randint(0, h - crop_size)
        rand_w = random.randint(0, w - crop_size)
        rand_h = (rand_h // min_divide) * min_divide
        rand_w = (rand_w // min_divide) * min_divide
        return rand_h, rand_w
    
    def get_train_img(self):
        train_rid = random.randint(0, len(self.train_list) - 1)
        train_img_input_path = os.path.join(self.opt.baseroot_train, self.train_list[train_rid] + self.opt.ablation_data + '.png')
        train_img_rawout_path = os.path.join(self.opt.baseroot_train, self.train_list[train_rid] + '_quadbayer_short.png')
        train_img_rgbout_path = os.path.join(self.opt.baseroot_train, self.train_list[train_rid] + '_rgb_gt.png')
        train_img_input = cv2.imread(train_img_input_path, -1)
        train_img_rawout = cv2.imread(train_img_rawout_path, -1)
        train_img_rgbout = cv2.imread(train_img_rgbout_path, -1)
        return train_img_input, train_img_rawout, train_img_rgbout
    
    def get_blur_patch(self):
        train_blur_patch_rid = random.randint(0, len(self.train_blur_patch_list) - 1)
        train_blur_patch_input_path = os.path.join(self.opt.baseroot_train_blur_patch, self.train_blur_patch_list[train_blur_patch_rid] + self.opt.ablation_data + '.png')
        train_blur_patch_rawout_path = os.path.join(self.opt.baseroot_train_blur_patch, self.train_blur_patch_list[train_blur_patch_rid] + '_quadbayer_short.png')
        train_blur_patch_rgbout_path = os.path.join(self.opt.baseroot_train_blur_patch, self.train_blur_patch_list[train_blur_patch_rid] + '_rgb_gt.png')
        train_blur_patch_input = cv2.imread(train_blur_patch_input_path, -1)
        train_blur_patch_rawout = cv2.imread(train_blur_patch_rawout_path, -1)
        train_blur_patch_rgbout = cv2.imread(train_blur_patch_rgbout_path, -1)
        return train_blur_patch_input, train_blur_patch_rawout, train_blur_patch_rgbout
    
    def get_salient_patch(self):
        train_salient_patch_rid = random.randint(0, len(self.train_salient_patch_list) - 1)
        train_salient_patch_input_path = os.path.join(self.opt.baseroot_train_salient_patch, self.train_salient_patch_list[train_salient_patch_rid] + self.opt.ablation_data + '.png')
        train_salient_patch_rawout_path = os.path.join(self.opt.baseroot_train_salient_patch, self.train_salient_patch_list[train_salient_patch_rid] + '_quadbayer_short.png')
        train_salient_patch_rgbout_path = os.path.join(self.opt.baseroot_train_salient_patch, self.train_salient_patch_list[train_salient_patch_rid] + '_rgb_gt.png')
        train_salient_patch_input = cv2.imread(train_salient_patch_input_path, -1)
        train_salient_patch_rawout = cv2.imread(train_salient_patch_rawout_path, -1)
        train_salient_patch_rgbout = cv2.imread(train_salient_patch_rgbout_path, -1)
        return train_salient_patch_input, train_salient_patch_rawout, train_salient_patch_rgbout
    
    def get_joint_patch(self):
        train_joint_patch_rid = random.randint(0, len(self.train_joint_patch_list) - 1)
        train_joint_patch_input_path = os.path.join(self.opt.baseroot_train_joint_patch, self.train_joint_patch_list[train_joint_patch_rid] + self.opt.ablation_data + '.png')
        train_joint_patch_rawout_path = os.path.join(self.opt.baseroot_train_joint_patch, self.train_joint_patch_list[train_joint_patch_rid] + '_quadbayer_short.png')
        train_joint_patch_rgbout_path = os.path.join(self.opt.baseroot_train_joint_patch, self.train_joint_patch_list[train_joint_patch_rid] + '_rgb_gt.png')
        train_joint_patch_input = cv2.imread(train_joint_patch_input_path, -1)
        train_joint_patch_rawout = cv2.imread(train_joint_patch_rawout_path, -1)
        train_joint_patch_rgbout = cv2.imread(train_joint_patch_rgbout_path, -1)
        return train_joint_patch_input, train_joint_patch_rawout, train_joint_patch_rgbout

    def __getitem__(self, index):

        # Number of images
        num_train_img, num_blur_patch, num_salient_patch, num_joint_patch \
            = self.opt.num_input[0], self.opt.num_input[1], self.opt.num_input[2], self.opt.num_input[3]

        # Read images
        train_img_input, train_img_rawout, train_img_rgbout = self.get_train_img()
        if num_blur_patch > 0:
            train_blur_patch_input, train_blur_patch_rawout, train_blur_patch_rgbout = self.get_blur_patch()
        if num_salient_patch > 0:
            train_salient_patch_input, train_salient_patch_rawout, train_salient_patch_rgbout = self.get_salient_patch()
        if num_joint_patch > 0:
            train_joint_patch_input, train_joint_patch_rawout, train_joint_patch_rgbout = self.get_joint_patch()

        # Extract patches
        input_list = []
        rawout_list = []
        rgbout_list = []
        for i in range(num_train_img):
            h, w = train_img_input.shape[:2]
            rand_h, rand_w = self.random_crop_start(h, w, self.opt.crop_size, 4)
            input_list.append(train_img_input[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size])
            rawout_list.append(train_img_rawout[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size])
            rgbout_list.append(train_img_rgbout[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :])
        if num_blur_patch > 0:
            input_list.append(train_blur_patch_input)
            rawout_list.append(train_blur_patch_rawout)
            rgbout_list.append(train_blur_patch_rgbout)
        if num_salient_patch > 0:
            input_list.append(train_salient_patch_input)
            rawout_list.append(train_salient_patch_rawout)
            rgbout_list.append(train_salient_patch_rgbout)
        if num_joint_patch > 0:
            input_list.append(train_joint_patch_input)
            rawout_list.append(train_joint_patch_rawout)
            rgbout_list.append(train_joint_patch_rgbout)

        # Post-process images
        input_processed = []
        rawout_processed = []
        rgbout_processed = []
        edgeout_processed = []
        for id in range(len(input_list)):
            
            ### build input images list
            img = input_list[id]
            # Normalization
            img = img.astype(np.float) / 16383.0
            # Add noises
            for pos in self.short_pos:
                img[pos[0]::2, pos[1]::2] /= 4
            img = DA.add_noise_raw(img, self.opt.noise_K, self.opt.noise_sigma)
            for pos in self.short_pos:
                img[pos[0]::2, pos[1]::2] *= 4
            img = np.clip(img, 0, 1)
            # Add gamma correction
            img = img ** (1 / 2.2)
            # To tensor
            input_processed.append(torch.from_numpy(img).float().unsqueeze(0).contiguous())

            ### build rawout images list
            img = rawout_list[id]
            # Normalization
            img = img.astype(np.float) / 16383.0
            # Add gamma correction
            img = img ** (1 / 2.2)
            # To tensor
            rawout_processed.append(torch.from_numpy(img).float().unsqueeze(0).contiguous())

            ### build rgbout / edgeout images list
            img = rgbout_list[id]
            # Normalization
            edge = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edge = DA.sobel(edge)
            edge = edge.astype(np.float) / 255.0
            img = img.astype(np.float) / 255.0
            # To tensor
            rgbout_processed.append(torch.from_numpy(img).float().permute(2, 0, 1).contiguous())
            edgeout_processed.append(torch.from_numpy(edge).float().unsqueeze(0).contiguous())
        
        # Concatenate
        for id2 in range(len(input_processed)):
            if id2 == 0:
                input_batch = input_processed[id2].unsqueeze(0)
                rawout_batch = rawout_processed[id2].unsqueeze(0)
                rgbout_batch = rgbout_processed[id2].unsqueeze(0)
                edgeout_batch = edgeout_processed[id2].unsqueeze(0)
            else:
                input_batch = torch.cat((input_batch, input_processed[id2].unsqueeze(0)), 0)
                rawout_batch = torch.cat((rawout_batch, rawout_processed[id2].unsqueeze(0)), 0)
                rgbout_batch = torch.cat((rgbout_batch, rgbout_processed[id2].unsqueeze(0)), 0)
                edgeout_batch = torch.cat((edgeout_batch, edgeout_processed[id2].unsqueeze(0)), 0)
            input_batch = input_batch.contiguous()
            rawout_batch = rawout_batch.contiguous()
            rgbout_batch = rgbout_batch.contiguous()
            edgeout_batch = edgeout_batch.contiguous()
        
        sample = {'input_batch': input_batch,
                  'rawout_batch': rawout_batch,
                  'rgbout_batch': rgbout_batch,
                  'edgeout_batch': edgeout_batch}

        return sample
    
    def __len__(self):
        return 10000

class QuadBayer2RGB_Valset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.blur_imglist = []
        self.sharp_imglist = []
        self.save_imglist = []

        # Build validation dataset
        imglist = get_heads_train(opt.baseroot_val)
        for i in range(len(imglist)):
            imgname = '%s%s_K%0.2f_sigma%0.1f.png' % (imglist[i], opt.ablation_data, opt.noise_K, opt.noise_sigma)
            self.blur_imglist.append(os.path.join(opt.baseroot_val, imgname))
            self.sharp_imglist.append(os.path.join(opt.baseroot_val, imglist[i] + '_rgb_gt.png'))
            self.save_imglist.append(imgname)

    def __getitem__(self, index):
        
        # Path of one image
        blur_img_path = self.blur_imglist[index]
        clean_img_path = self.sharp_imglist[index]
        save_img_path = self.save_imglist[index]

        in_img = cv2.imread(blur_img_path, -1)
        RGBout_img = cv2.imread(clean_img_path, -1)

        # Input images
        in_img = in_img.astype(np.float) / 16383.0
        in_img = in_img ** (1 / 2.2)
        in_img = torch.from_numpy(in_img).float().unsqueeze(0).contiguous()

        # Target images
        RGBout_img = RGBout_img.astype(np.float) / 255.0
        RGBout_img = torch.from_numpy(RGBout_img).float().permute(2, 0, 1).contiguous()

        return in_img, RGBout_img, save_img_path
    
    def __len__(self):
        return len(self.blur_imglist)

if __name__ == "__main__":

    baseroot_test = 'F:\\QuadBayer Deblur\\data\\test\\test1'
    imglist = get_raw_test(baseroot_test)
    print(imglist)
