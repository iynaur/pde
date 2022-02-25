from itertools import repeat
from math import *
import random
from typing import NoReturn
import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('Qt5Agg')

import cv2
import sys, os

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# a * du2 / d2x + b * du2 / d2y - c *u + d = 0

# i, j for ith row, jth col
crow = 300
ccol = 300
if 1:
    testdir = './test'
    flist = os.listdir(testdir)
    img = cv2.imread(os.path.join(testdir, flist[0]), cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (5, 5), 3)
    resize = 1
    if resize :
        img = cv2.resize(img, (ccol + 2, crow + 2))
    else:
        img = img[:ccol + 2, :crow+2]
        crow = img.shape[0] - 2
        ccol = img.shape[1] - 2
    top, bottom, left, right = 4,4,4,4
    # img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
    crow = img.shape[0] - 2
    ccol = img.shape[1] - 2
else:
    img = np.zeros((crow + 2, ccol + 2))
    for row in range(crow + 2):
        for col in range(ccol + 2):
            img[row, col] = 2*4*pi**2/crow/ccol*cos((row-1)/(crow)*2*pi)*cos((col-1)/(ccol)*2*pi)
    pass
    # img /= np.max(img)

# cv2.imshow("", img)
# cv2.waitKey()
# plt.imshow(img)
# plt.show()

def ga(i, j):
    return 1

def gb(i, j):
    return 1

def gc(i, j):
    return 0
    return 2*4*pi**2/crow/ccol

def gdx(i, j):
    # return fabs(i - crow/2) + fabs(j - ccol/2)
    return \
        -((float)(img[i+1,j]) - img[i+1,j+2])/2
    #     (i- crow / 2)  \
    # +  ccol**2/((j- ccol / 2) **2 +1)
    # return 1

def gdy(i, j):
    # return fabs(i - crow/2) + fabs(j - ccol/2)
    return \
        -((float)(img[i,j+1]) - img[i+2,j+1])/2

def d(i, j):
    # return fabs(i - crow/2) + fabs(j - ccol/2)
    return \
        (float)(img[i+1,j+1])

a = np.ones((crow, ccol)) * 2
b = np.ones((crow, ccol))
c = np.ones((crow, ccol)) / 300
dx = np.zeros((crow, ccol))
dy = np.ones((crow, ccol))
f = np.ones((crow, ccol))
mats = [dx, f]
funcs = [gdx, d]

# for mat, func in zip(mats, funcs):
#     for row in range(crow):
#         for col in range(ccol):
#             mat[row, col] = func(row, col)

import numba
from numba import threading_layer, set_num_threads, config
from funcy import print_durations

# config.THREADING_LAYER = 'tbb'

set_num_threads(6)

@numba.jit(nogil=True, nopython = True)
def proc2(i, Lap_u, u):
    w = 1.0
    # print(w)
    # for j in numba.prange(0, ccol):
    for j in range(0, ccol):
        # for k in range(j**2):
        #     Lap_u[i,j] += j
        nu = b[i, j]*(u[(i+1)%crow, j] + u[i-1, j]) \
        + a[i,j]*(u[i, (j+1)%ccol] + u[i, j-1]) + dx[i, j]
        nu /= (2*a[i, j] + 2*b[i, j] + c[i, j])
        diff = nu - u[i, j]
        Lap_u[i,j] = u[i, j] + w*diff

@print_durations()
@numba.jit(nogil=True, nopython = True)
def Nxt_solver(iter = 100, ):
    w = 1.0
    # 对中间点的五点法处理
    # crop 1 pixel
    Lap_u = np.zeros((crow, ccol))
    u = np.zeros((crow, ccol))
    # diff = np.zeros((crow, ccol))
    # diffs = []



    for i in range(iter):
        # u[:] = Lap_u
        if 0: # ThreadPoolExecutor slower for small jobs
            with ThreadPoolExecutor(max_workers=1) as executor:
                _ = executor.map(proc2, range(0, crow), repeat(Lap_u), repeat(u))
            # _ = list(_)
            pass
        else:
            for i in range(0, crow):
            # for i in numba.prange(0, crow):
                proc2(i, Lap_u, u)
                continue
                for j in range(0, ccol):
                # for k in range(j**2):
                #     Lap_u[i,j] += j
                    nu = b[i, j]*(u[(i+1)%crow, j] + u[i-1, j]) \
                    + a[i,j]*(u[i, (j+1)%ccol] + u[i, j-1]) + dx[i, j]
                    nu /= (2*a[i, j] + 2*b[i, j] + c[i, j])
                    diff = nu - u[i, j]
                    Lap_u[i,j] = u[i, j] + w*diff
            #
        u[:] = Lap_u
        continue

        # Lap_u = Lap_u / (2*a + 2*b + c)
        # # return Lap_u # w = 1
        # ndiff = Lap_u - u
        # crop = 4
        # diff = np.max(np.fabs(ndiff[crop:-crop, crop:-crop]))
        # diffs.append(diff)
        # # u, Lap_u = Lap_u, u

    return Lap_u, None

@numba.jit(nogil=True, nopython = True)
def proc(i, u, even, w):
    # assert( i % 2 == 0 and ccol % 2 == 0)
    madif = 0.0
    for j in range((even+i+1)%2 + 1, ccol-1, 2):
        # if fabs(i - crow//2) <= crow/4 and fabs(j - ccol//2) ==0:
        #     u[i, j] =  1000
        #     continue

        nu = b[i, j]*(u[(i+1)%crow, j] + u[i-1, j]) \
        + a[i,j]*(u[i, (j+1)%ccol] + u[i, j-1]) + dx[i, j]
        nu /= (2*a[i, j] + 2*b[i, j] + c[i, j])
        diff = nu - u[i, j]
        madif = max(madif, fabs(diff))
        u[i, j] += w*diff
    return madif

@print_durations()
@numba.jit(nogil=True, parallel=dict(prange=True, fusion=False))
def SOR_solver(iter = 100, w = 1.0,):
    # 对中间点的五点法处理
    # crop 1 pixel
    u = np.zeros((crow, ccol))
    # diff = np.zeros((crow, ccol))
    # diffs = []
    residuals = np.zeros(iter)


            # # boundray
            # if i == 0 or i == crow-1 or j ==0 or j==ccol-1:
            #     u[i, j] = 0
            # if i == 0 and j==0:
            #     u[i, j] = 1

    for row in range(crow):
        u[row, ccol-1] = random.random() *2-1
        u[row, 0] = -1
    for col in range(ccol):
        u[crow-1, col] = col /ccol * 2 - 1
        u[0, col] = col /ccol * 2 - 1
    u[crow//2, 0] = -1
    madiff = np.zeros(crow)
    for ii in range(iter):
        # madiff = np.zeros(crow)
        madiff *= 0
        # u[:] = Lap_u
        if False:
            with ThreadPoolExecutor(max_workers=4) as executor:
                _ = executor.map(proc, range(0, crow), repeat(u), repeat(0))
            with ThreadPoolExecutor(max_workers=4) as executor:
                _ = executor.map(proc, range(0, crow), repeat(u), repeat(1))
            # _ = list(_)
        else:
            # numba.prange
            if False:
                for i in range(1, crow-1):
                    madiff[i] =  proc(i, u, 0, w)
                for i in range(1, crow -1):
                    madiff[i] = max(madiff[i], proc(i, u, 1, w))
            else:
                for i in numba.prange(1, crow-1):
                    madiff[i] =  proc(i, u, 0, w)
                for i in numba.prange(1, crow -1):
                    madiff[i] = max(madiff[i], proc(i, u, 1, w))
            #
        residuals[ii] = np.max(madiff)
        continue

        Lap_u = Lap_u / (2*a + 2*b + c)
        # return Lap_u # w = 1
        ndiff = Lap_u - u
        crop = 4
        diff = np.max(np.fabs(ndiff[crop:-crop, crop:-crop]))
        diffs.append(diff)
        # u, Lap_u = Lap_u, u

    return u, residuals


def Nxt_SOR(u, d, w = 1.0):
    # 对中间点的五点法处理
    # crop 1 pixel
    ogu = u.copy()
    Lap_u = u #.copy()
    # for i in range(1, row-1):
    #     for j in range(1, col - 1):
    for i in range(0, crow):
        for j in range(0, ccol):
            Lap_u[i,j] = b[i, j]*(u[(i+1)%crow, j] + Lap_u[i-1, j]) \
            + a[i,j]*(u[i, (j+1)%ccol] + Lap_u[i, j-1]) + d[i, j]

            Lap_u[i,j] /= (2*a[i,j] + 2*b[i,j] + c[i,j])
    # 对边界点处理

    # #  up and down edge
    # for i in range(1, ccol-1):
    #     Lap_u[0, i] = 2*Lap_u[1,i] - Lap_u[2,i]
    #     Lap_u[ - 1, i] = 2*Lap_u[ - 2,i] - Lap_u[-3,i]
    # # left and right
    # for i in range(1, crow-1):
    #     Lap_u[i, 0] = 2*Lap_u[i, 1] - Lap_u[i, 2]
    #     Lap_u[i, -1] = 2*Lap_u[i, -2] - Lap_u[i, -3]

    # # corners
    # Lap_u[0,0] = Lap_u[0,1] + Lap_u[1,0] - Lap_u[1,1]
    # Lap_u[0,-1] = Lap_u[0,-2] + Lap_u[1,-1] - Lap_u[1,-2]
    # Lap_u[-1,0] = Lap_u[-1,1] + Lap_u[-2,0] - Lap_u[-2,1]
    # Lap_u[-1,-1] = Lap_u[-1,-2] + Lap_u[-2,-1] - Lap_u[-2,-2]

    # 略

    # return Lap_u # w = 1
    ndiff = Lap_u - ogu
    crop = 4
    diff = np.max(np.fabs(ndiff[crop:-crop, crop:-crop]))
    return ogu + w * ndiff, diff

def fftSolver(f = None, dx= None):
    assert((f is None) ^ (dx is None))
    if f is not None: # input raw f
        m = f.shape[0]
        n = f.shape[1]
        cf = np.fft.fft2(f)
        cd = np.zeros(f.shape, dtype = complex)
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                # this is crucial
                ni = min(i, m-i)
                nj = min(j, n-j)
                rq = j if (j < (n-j)) else j - n
                if fabs(rq) > n/2 * 0.5:
                    # rq = 0
                    pass
                cd[i, j] = 2 *pi * rq * (0+1j) / n *cf[i,j]

                cf[i, j] = 2 *pi * rq * (0+1j) / n / \
                (a[i,j]*(2*pi*ni/m)**2 + b[i,j]*(2*pi*nj/n)**2 + c[i,j]) * cf[i,j]

        # rfft = np.fft.ifft2(cd)
        # plt.imshow(np.real(rfft))
        # plt.show()
        u = np.fft.ifft2(cf)
        return np.real(u)
    I = 0+1j

    f = dx
    cu = np.zeros(f.shape, dtype = complex)
    cf = np.fft.fft2(f)
    m = f.shape[0]
    n = f.shape[1]
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            # this is crucial
            ni = min(i, m-i)
            nj = min(j, n-j)
            cu[i, j] = 1 / \
            (a[i,j]*(2*pi*ni/m)**2 + b[i,j]*(2*pi*nj/n)**2 + c[i,j]) * cf[i,j]
    # cu[0][0] = 0
    u = np.fft.ifft2(cu)
    return np.real(u)



if __name__ == "__main__":

    diff_cmp = []
    diffs_sor_cmp = []
    for w in np.linspace(1.2, 1.2, num=1): # w > 1 may converge then diverge
        diffs = []
        diffs_sor = []
        ux = np.zeros((crow, ccol))
        uy = np.zeros((crow, ccol))
        for iter in range(0):
            # ux, diff= Nxt(ux, dx, w = 1.0)
            uy, _ = Nxt_SOR(uy, dx, w = 1.5)
            # diffs.append(diff)
            diffs_sor.append(_)
        # ux, diffs = Nxt_solver( 360)
        for r in range(1):
            uy, resid = SOR_solver(5000, w = 1.8 )
            diff_cmp.append(resid)
        diffs_sor_cmp.append(diffs_sor)
    print("Threading layer chosen: %s" % threading_layer())

    X, Y = np.meshgrid(range(ccol), range(crow))
    if 0:
        crop = 10
        scale = np.max(ux[crop:-crop, crop:-crop])*1.5

        plt.imshow(ux**2 + uy**2)
        plt.quiver(X, Y, ux/scale, uy/scale, angles='xy', scale_units='xy', scale=1)
        plt.show()


    # uz = fftSolver(f = f) # not acurate for input image not smooth at period bondray
    ud = fftSolver(dx = dx)

    toplot = [ uy, ]
    for i, u in enumerate(toplot):
        cnt = len(toplot)
        plt.subplot(1, cnt+1, i+1)
        bu = (np.real(u))
        # bu = np.append(bu, bu, 0)
        # bu = np.append(bu, bu, 1)
        if 1: fig = plt.imshow(bu)
        else: fig = plt.imshow(np.real(u - uy))

        # ax3 = plt.axes(projection='3d')
        # plt.plot_surface(X, Y, u, cmap='rainbow')
        # plt.contour(X, Y, u, colors='black')   #等高线图，要设置offset，为Z的最小值
        # b = plt.contour(X, Y, u, 3, colors='black', linewidths=1, linestyles='solid')
        plt.colorbar(fig)
        fig.set_cmap('jet') # 'plasma' or 'viridis'
    # plt.show()
    # exit(0)
    plt.subplot(1, cnt+1, cnt+1)
    for diffs in diff_cmp:
        plt.plot(diffs)
    # for diffs in diffs_sor_cmp:
    #     plt.plot(diffs)
    plt.show()