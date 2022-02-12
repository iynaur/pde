from typing import NoReturn
import numpy as np
import matplotlib.pyplot as plt

import cv2
import sys, os



# a * du2 / d2x + b * du2 / d2y - c *u + d = 0

# i, j for ith row, jth col
crow = 200
ccol = 200

testdir = './test'
flist = os.listdir(testdir)
img = cv2.imread(os.path.join(testdir, flist[0]), cv2.IMREAD_GRAYSCALE)

resize = 0
if resize :
    img = cv2.resize(img, (ccol + 2, crow + 2))
else:
    # img = img[:ccol + 2, :crow+2]
    crow = img.shape[0] - 2
    ccol = img.shape[1] - 2

def ga(i, j):
    return 1

def gb(i, j):
    return 1

def gc(i, j):
    return 1

def gdx(i, j):
    return \
        (float)(img[i,j]) - img[i,j+2]
    #     (i- crow / 2)  \
    # +  ccol**2/((j- ccol / 2) **2 +1)
    # return 1

def gdy(i, j):
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

    # coners
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

if __name__ == "__main__":

    diff_cmp = []
    for w in np.linspace(0.5, 0.99, num=2): # w > 1 may converge then diverge
        diffs = []
        ux = np.zeros((crow, ccol))
        uy = np.zeros((crow, ccol))
        for iter in range(40):
            ux, diff= Nxt(ux, dx, w)
            uy, _ = Nxt(uy, dy, w)
            diffs.append(diff + _)
        diff_cmp.append(diffs)

    crop = 10
    scale = np.max(ux[crop:-crop, crop:-crop])*1.5
    X, Y = np.meshgrid(range(ccol), range(crow))
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
    plt.show()