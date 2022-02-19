from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from math import *
import cmath
from typing import NoReturn
import numpy as np
import matplotlib.pyplot as plt

import cv2
import sys, os
from multiprocessing import Pool
import numba
from itertools import repeat


def ifft_ups(fft, ups):

    crow, ccol = fft.shape
    om = cmath.exp(2*pi/crow * (0+1j) )
    on = cmath.exp(2*pi/ccol * (0+1j) )
    rfft = np.zeros((crow*ups, ccol* ups), dtype=np.complex_)

    @numba.jit(nogil=True)
    def proc(ii, rfft):
        i = ii/ups
        # rfft = np.zeros(ccol* ups, dtype=np.complex_)
        for jj in range(0, ccol* ups, 1):
            j = jj/ups
            for p in range(crow):
                for q in range(ccol):
                    # better interpolate
                        nnp = p if p < crow -p else p - crow
                        nq = q if q<  ccol - q else q - ccol
                        rfft[ii, jj] += fft[p, q] * om ** (nnp*i) * on ** (nq*j)
        return 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        ite = executor.map(proc, range(0, crow* ups, 1), repeat(rfft))
        # rfft = np.array(list(ite))
    # plt.imshow(np.real(rfft))
    # plt.show(block=1)
    return rfft

# @numba.jit(nogil=True)
def partialx_fft(f, ups = 1):
    crow, ccol = f.shape
    fft = np.fft.fft2(f)
    for p in range(crow):
        for q in range(ccol):
            rq = q if (q < (ccol-q)) else q - ccol
            if fabs(rq) > ccol/2 * 0.5:
                # rq = 0
                pass
            # rp = p
            fft[p, q] = fft[p, q] * 2 *pi * rq * (0+1j) / crow
    rfft = np.fft.ifft2(fft)
    # plt.imshow(np.real(rfft))
    # plt.show(block=1)
    return fft