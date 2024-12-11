import cv2
import numpy as np

short_path = './test_results/QRNet__K0.25__sigma5.0__data4000__loss4/DEBLUR_scene02_ref.RAWMIPI.png'
long_path = './test_results/QRNet__K0.25__sigma5.0__data4000__loss4/DEBLUR_scene02_02.RAWMIPI.png'

short_img = cv2.imread(short_path)
long_img = cv2.imread(long_path)
fusion_img = np.zeros_like(short_img).astype(np.uint8)

short_pos = [[0,0], [1,1]]
long_pos = [[0,1], [1,0]]

for pos in short_pos:
    fusion_img[pos[0]::2, pos[1]::2] = short_img[pos[0]::2, pos[1]::2]
for pos in long_pos:
    fusion_img[pos[0]::2, pos[1]::2] = long_img[pos[0]::2, pos[1]::2]

cv2.imwrite('save.png', fusion_img)
