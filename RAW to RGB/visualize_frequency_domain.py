import cv2
import numpy as np
from matplotlib import pyplot as plt

#imgpath = './val_results/QRNet__K0.25__sigma5.0__data4000__loss4/1_quadbayer_ls_K0.25_sigma5.0.png'
imgpath = '../data/val/1_rgb_gt.png'

img = cv2.imread(imgpath, 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

cv2.imwrite('save.png', magnitude_spectrum)

plt.subplot(121), plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
