# QRNet

## 1 Brief Introduction

QRNet is a state-of-the-art neural network that transforms the Quad-Bayer pattern into sharp and clean RGB images. Two examples and comparisons with existing methods are illustrated below:

<img src='img/show1.png' width=1000>

<img src='img/show2.png' width=1000>

<img src='img/show3.png' width=1000>

## 2 Train

The training is done on 4 NVIDIA Titan Xp GPUs, where 4 batches run on each GPU card. The training can be also performed on a single GPU card:
```bash
python train.py --yaml_path options/qrnet_raw1_data4000_loss4.yaml
```

## 3 Validation

For validation on QR dataset, please run:
```bash
python validation.py --yaml_path options/qrnet_raw1_data4000_loss4.yaml
```

## 4 Dataset

### 4.1 Data Synthesis

We propose a new dataset called QR dataset, which includes 731 pairs of Quad-Bayer and RGB images. The data synthesis and capturing process is shown below:

<img src='img/data1.png' width=1000>

<img src='img/data2.png' width=1000>

### 4.2 Validation Data Synthesis (+noise)

To build validation tuples, we add physical noises to different types of input data as below (the brightness of input Quad-Bayer is enhanced for visualization):

<img src='img/dataval1.png' width=1000>

<img src='img/dataval2.png' width=1000>
