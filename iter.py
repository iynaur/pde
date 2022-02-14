from math import *
from typing import NoReturn
import numpy as np
import matplotlib.pyplot as plt

import cv2
import sys, os



# a * du2 / d2x + b * du2 / d2y - c *u + d = 0

# i, j for ith row, jth col
crow = 200
ccol = 200
if 0:
    testdir = './test'
    flist = os.listdir(testdir)
    img = cv2.imread(os.path.join(testdir, flist[0]), cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (5, 5), 3)
    resize = 0
    if resize :
        img = cv2.resize(img, (ccol + 2, crow + 2))
    else:
        # img = img[:ccol + 2, :crow+2]
        crow = img.shape[0] - 2
        ccol = img.shape[1] - 2
else:
    img = np.zeros((crow + 2, ccol + 2))
    for row in range(crow + 2):
        for col in range(ccol + 2):
            img[row, col] = fabs(row - crow/2 -1) + fabs(col - ccol/2 -1)
    pass
    img /= np.max(img)

# cv2.imshow("", img)
# cv2.waitKey()

def ga(i, j):
    return 1

def gb(i, j):
    return 1

def gc(i, j):
    return 1

def gdx(i, j):
    # return fabs(i - crow/2) + fabs(j - ccol/2)
    return \
        (float)(img[i,j]) - img[i,j+2]
    #     (i- crow / 2)  \
    # +  ccol**2/((j- ccol / 2) **2 +1)
    # return 1

def gdy(i, j):
    # return fabs(i - crow/2) + fabs(j - ccol/2)
    return \
        (float)(img[i,j]) - img[i+2,j]

a = np.zeros((crow, ccol))
b = np.zeros((crow, ccol))
c = np.zeros((crow, ccol))
dx = np.zeros((crow, ccol))
dy = np.zeros((crow, ccol))

mats = [a,b,c,dx, dy]
funcs = [ga, gb, gc, gdx, gdy]

for mat, func in zip(mats, funcs):
    for row in range(crow):
        for col in range(ccol):
            mat[row, col] = func(row, col)

def Nxt(u, d, w = 1.0):
    # 对中间点的五点法处理
    # crop 1 pixel
    Lap_u = np.zeros((crow, ccol))
    Lap_u[1:-1, 1:-1] = b[1:-1, 1:-1]*(u[2:, 1:-1] + u[:-2, 1:-1]) \
            + a[1:-1, 1:-1]*(u[1:-1, 2:] + u[1:-1, :-2]) + d[1:-1, 1:-1]
    # 对边界点处理

    #  up and down edge
    for i in range(1, ccol-1):
        Lap_u[0, i] = 2*Lap_u[1,i] - Lap_u[2,i]
        Lap_u[ - 1, i] = 2*Lap_u[ - 2,i] - Lap_u[-3,i]
    # left and right
    for i in range(1, crow-1):
        Lap_u[i, 0] = 2*Lap_u[i, 1] - Lap_u[i, 2]
        Lap_u[i, -1] = 2*Lap_u[i, -2] - Lap_u[i, -3]

    # corners
    Lap_u[0,0] = Lap_u[0,1] + Lap_u[1,0] - Lap_u[1,1]
    Lap_u[0,-1] = Lap_u[0,-2] + Lap_u[1,-1] - Lap_u[1,-2]
    Lap_u[-1,0] = Lap_u[-1,1] + Lap_u[-2,0] - Lap_u[-2,1]
    Lap_u[-1,-1] = Lap_u[-1,-2] + Lap_u[-2,-1] - Lap_u[-2,-2]

    # 略
    Lap_u = Lap_u / (2*a + 2*b + c)
    # return Lap_u # w = 1
    ndiff = Lap_u - u
    crop = 4
    diff = np.max(np.fabs(ndiff[crop:-crop, crop:-crop]))
    return u + w * ndiff, diff

def Nxt_SOR(u, d, w = 1.0):
    # 对中间点的五点法处理
    # crop 1 pixel
    Lap_u = u.copy()
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
    ndiff = Lap_u - u
    crop = 4
    diff = np.max(np.fabs(ndiff[crop:-crop, crop:-crop]))
    return u + w * ndiff, diff

if __name__ == "__main__":

    diff_cmp = []
    diffs_sor_cmp = []
    for w in np.linspace(0.5, 0.99, num=2): # w > 1 may converge then diverge
        diffs = []
        diffs_sor = []
        ux = np.zeros((crow, ccol))
        uy = np.zeros((crow, ccol))
        for iter in range(30):
            ux, diff= Nxt(ux, dx, w)
            uy, _ = Nxt_SOR(uy, dy, w)
            diffs.append(diff)
            diffs_sor.append(_)
        diff_cmp.append(diffs)
        diffs_sor_cmp.append(diffs_sor)

    
    X, Y = np.meshgrid(range(ccol), range(crow))
    if 0:
        crop = 10
        scale = np.max(ux[crop:-crop, crop:-crop])*1.5
        
        plt.imshow(ux**2 + uy**2)
        plt.quiver(X, Y, ux/scale, uy/scale, angles='xy', scale_units='xy', scale=1)
        plt.show()



    for i, u in enumerate([ux, uy]):
        plt.subplot(1, 3, i+1)

        fig = plt.imshow(u)

        # ax3 = plt.axes(projection='3d')
        # plt.plot_surface(X, Y, u, cmap='rainbow')
        plt.contour(X, Y, u, colors='black')   #等高线图，要设置offset，为Z的最小值
        # b = plt.contour(X, Y, u, 3, colors='black', linewidths=1, linestyles='solid')
        plt.colorbar(fig)
        fig.set_cmap('jet') # 'plasma' or 'viridis'
    # plt.show()
    # exit(0)
    plt.subplot(133)
    for diffs in diff_cmp:
        plt.plot(diffs)
    for diffs in diffs_sor_cmp:
        plt.plot(diffs)
    plt.show()