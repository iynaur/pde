
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from math import *
import cmath
from typing import NoReturn
import numpy as np
import matplotlib.pyplot as plt

import cv2
import sys, os
from multiprocessing import Pool


# a * du2 / d2x + b * du2 / d2y - c *u + d = 0

# i, j for ith row, jth col
crow = 40
ccol = 40
if 1:
    testdir = './test'
    flist = os.listdir(testdir)
    img = cv2.imread(os.path.join(testdir, flist[0]), cv2.IMREAD_GRAYSCALE)

    resize = 1
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


# plt.imshow(img)
# plt.show()

fft = np.fft.fft2(img)
rfft = np.fft.ifft2(fft)

# plt.imshow(np.abs(fft))
# plt.show()
# rfft = np.fft.ifft2(fft)
# plt.imshow(np.real(rfft))
# plt.show()

# print(fft)
# for p in range(crow):
#     for q in range(ccol):
#         rq = q if (q < (ccol-q)) else q - ccol
#         if fabs(rq) > ccol/2 * 0.5:
#             # rq = 0
#             pass
#         # rp = p
#         fft[p, q] = fft[p, q] * 2 *pi * rq * (0+1j) / crow
# rfft = np.fft.ifft2(fft)
# plt.imshow(np.real(rfft))
# plt.show()
# exit()
ups = 4
rfft = np.zeros((crow * ups, ccol * ups), np.complex)
om = cmath.exp(2*pi/crow * (0+1j) )
on = cmath.exp(2*pi/ccol * (0+1j) )
N = 8
validrow = [i for i in range(crow) if fabs(i - crow/2) > crow/8]
validcol = [i for i in range(ccol) if fabs(i - ccol/2) > ccol/8]

import numba

@numba.jit(nogil=True)
def proc(ii):
    i = ii/ups
    rfft = np.zeros(ccol* ups, dtype=complex)
    for jj in range(0, ccol* ups, 1):
        j = jj/ups
        for p in range(crow):
            for q in range(ccol):
                # better interpolate
                    nnp = p if p < crow -p else p - crow
                    nq = q if q<  ccol - q else q - ccol
                    rfft[jj] += fft[p, q] * om ** (nnp*i) * on ** (nq*j)
    return rfft

if __name__ == "__main__":
    if 1:
        with ThreadPoolExecutor(max_workers=4) as executor:
            future = executor.map(proc, range(0, crow* ups, 1))
            # result = [f.result() for f in future]
            # print(result)
            #     result = pool.map(proc, range(0, crow* ups, 1))
            #     pool.close()
            #     pool.join()

        rfft  = np.array(list(future))


    else:
        for ii in range(0, crow* ups, 1):
            i = ii/ups
            for jj in range(0, ccol* ups, 1):
                j = jj/ups
                for p in range(crow):
                    for q in range(ccol):
                        # better interpolate
                        nnp = p if p < crow -p else p - crow
                        nq = q if q<  ccol - q else q - ccol
                        rfft[ii, jj] += fft[p, q] * om ** (nnp*i) * on ** (nq*j)
    plt.imshow(np.real(rfft/crow/ccol))
    plt.show()

    plt.imshow(np.imag(rfft/crow/ccol))
    plt.show()