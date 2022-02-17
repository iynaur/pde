
from math import *
import cmath
from typing import NoReturn
import numpy as np
import matplotlib.pyplot as plt

import cv2
import sys, os



# a * du2 / d2x + b * du2 / d2y - c *u + d = 0

# i, j for ith row, jth col
crow = 20
ccol = 50
if 1:
    testdir = './test'
    flist = os.listdir(testdir)
    img = cv2.imread(os.path.join(testdir, flist[0]), cv2.IMREAD_GRAYSCALE)

    resize = 1
    if resize :
        img = cv2.resize(img, (ccol, crow))
    else:
        # img = img[:ccol + 2, :crow+2]
        crow = img.shape[0]
        ccol = img.shape[1]
    img = cv2.GaussianBlur(img, (5, 5), 3)
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

# print(fft)

rfft = np.zeros((crow, ccol), np.complex)
om = cmath.exp(2*pi/crow * (0+1j) )
on = cmath.exp(2*pi/ccol * (0+1j) )
N = 8
validrow = [i for i in range(crow) if fabs(i - crow/2) > crow/8]
validcol = [i for i in range(ccol) if fabs(i - ccol/2) > ccol/8]
for i in range(crow):
    for j in range(ccol):
        for p in validrow:
            for q in validcol:

                    rfft[i, j] += fft[p, q] * om ** (p*i) * on ** (q*j)


plt.imshow(np.real(rfft/crow/ccol - img))
plt.show()