
from math import *
import cmath
from typing import NoReturn
import numpy as np
import matplotlib.pyplot as plt

import cv2
import sys, os



# a * du2 / d2x + b * du2 / d2y - c *u + d = 0

# i, j for ith row, jth col
crow = 200
ccol = 200
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
    img = np.zeros((crow + 2, ccol + 2))
    for row in range(crow + 2):
        for col in range(ccol + 2):
            img[row, col] = fabs(row - crow/2 -1) + fabs(col - ccol/2 -1)
    pass
    img /= np.max(img)

plt.imshow(img)
plt.show()

fft = np.fft.fft2(img)
rfft = np.fft.ifft2(fft)

plt.imshow(np.abs(fft))
plt.show()
rfft = np.fft.ifft2(fft)
plt.imshow(np.abs(rfft))
plt.show()

# print(fft)
for p in range(crow):
    for q in range(ccol):
        fft[p, q] = fft[p, q] * 2 *pi * p * (0+1j) / crow
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