
from math import *
import cmath
from typing import NoReturn
import numpy as np
import matplotlib.pyplot as plt

import cv2
import sys, os



# a * du2 / d2x + b * du2 / d2y - c *u + d = 0

# i, j for ith row, jth col
crow = 50
ccol = 50
if 1:
    testdir = './test'
    flist = os.listdir(testdir)
    img = cv2.imread(os.path.join(testdir, flist[0]), cv2.IMREAD_GRAYSCALE)

    resize = 0
    if resize :
        img = cv2.resize(img, (ccol, crow))
    else:
        img = img[:crow , :ccol]
        # crow = img.shape[0]
        # ccol = img.shape[1]
    # img = cv2.GaussianBlur(img, (5, 5), 3)
else:
    abl = [[1,1], [3,5], [4,7]]
    img = np.zeros((crow, ccol))
    for row in range(crow):
        for col in range(ccol ):
            for ab in abl:
                img[row, col] += 2*4*pi**2/crow/ccol*cos(ab[0]*row/(crow)*2*pi)*cos(ab[1]*col/(ccol)*2*pi)


plt.imshow(img)
plt.show()

fft = np.fft.fft2(img)
rfft = np.fft.ifft2(fft)

plt.imshow(np.abs(fft))
plt.show()
rfft = np.fft.ifft2(fft)
plt.imshow(np.real(rfft))
plt.show()

# print(fft)
for p in range(crow):
    for q in range(ccol):
        rq = q if (q < (ccol-q)) else q - ccol
        if fabs(rq) > ccol/2 * 0.5:
            rq = 0
        # rp = p
        fft[p, q] = fft[p, q] * 2 *pi * rq * (0+1j) / crow
rfft = np.fft.ifft2(fft)
plt.imshow(np.real(rfft))
plt.show()
exit()
rfft = np.zeros((crow, ccol), np.complex)
om = cmath.exp(2*pi/crow * (0+1j) )
on = cmath.exp(2*pi/ccol * (0+1j) )
N = 8
validrow = [i for i in range(crow) if fabs(i - crow/2) > crow/8]
validcol = [i for i in range(ccol) if fabs(i - ccol/2) > ccol/8]
for i in range(crow):
    for j in range(ccol):
        for p in range(crow):
            for q in range(ccol):

                    rfft[i, j] += fft[p, q] * om ** (p*i) * on ** (q*j) * 2 *pi * p * (0+1j) / crow


plt.imshow(np.real(rfft/crow/ccol))
plt.show()