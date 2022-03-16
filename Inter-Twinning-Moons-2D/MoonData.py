import math
import numpy as np
import torch


def moondata(N,d,w,r,theta,bias):

    N1     = 10 * N
    w2     = w / 2
    done   = True
    data1  = np.empty(0)
    data2  = np.empty(0)
    angle  = math.radians(theta)
    pointx = abs(r) / 2
    pointy = abs(d) / 2

    while done:

        tmp_x1  = 2 * (r + w2) * (np.random.random([N1, 1]) - 0.5)
        tmp_x2  = 2 * (r + w2) * (np.random.random([N1, 1]) - 0.5)
        tmp_y1  = (r + w2) * np.random.random([N1, 1])
        tmp_y2  = (r + w2) * np.random.random([N1, 1])
        tmp1    = np.concatenate((tmp_x1, tmp_y1), axis=1)
        tmp2    = np.concatenate((tmp_x2, tmp_y2), axis=1)
        tmp_ds1 = np.sqrt(tmp_x1 * tmp_x1 + tmp_y1 * tmp_y1)
        tmp_ds2 = np.sqrt(tmp_x2 * tmp_x2 + tmp_y2 * tmp_y2)

        idx1 = np.logical_and(tmp_ds1 > (r - w2), tmp_ds1 < (r + w2))
        idx2 = np.logical_and(tmp_ds2 > (r - w2), tmp_ds2 < (r + w2))
        idx1 = (idx1.nonzero())[0]
        idx2 = (idx2.nonzero())[0]

        if data1.shape[0] == 0:
            data1 = tmp1.take(idx1, axis=0)
        else:
            data1 = np.concatenate((data1, tmp1.take(idx1, axis=0)), axis=0)
        if data2.shape[0] == 0:
            data2 = tmp2.take(idx2, axis=0)
        else:
            data2 = np.concatenate((data2, tmp2.take(idx2, axis=0)), axis=0)
        if data1.shape[0] >= N or data2.shape[0] >= N:
            done = False

    db_moon1 = data1[0:N, :]
    data_t1  = np.empty([N, 2])
    data_t1[:, 0] = data1[0:N, 0] + r
    data_t1[:, 1] = -data1[0:N, 1] - d
    db_moon1 = np.concatenate((db_moon1, data_t1), axis=0)

    db_moon2 = data2[0:N, :]
    data_t2 = np.empty([N, 2])
    data_t2[:, 0] = data2[0:N, 0] + r
    data_t2[:, 1] = -data2[0:N, 1] - d
    db_moon2 = np.concatenate((db_moon2, data_t2), axis=0)

    label_S1 = np.zeros((N,1))
    label_S2 = np.ones((N,1))

    T_x = (db_moon2[0:2*N,0] - pointx) * math.cos(angle) - (db_moon2[0:2*N,1] - pointy) * math.sin(angle) + pointx
    T_y = (db_moon2[0:2*N,0] - pointx) * math.sin(angle) + (db_moon2[0:2*N,1] - pointy) * math.cos(angle) + pointy
    T   = np.zeros((2*N,2))

    T[0:2*N,0] = T_x + bias
    T[0:2*N,1] = T_y + bias


    X_S = db_moon1
    Y_S = np.concatenate((label_S1, label_S2), axis=0)
    X_T = T

    np.savez('moon_randoms.npz',x_s = X_S, y_s = Y_S, x_t = X_T)

def dataload(mode,parameters):

    if mode == 0:
        data = np.load('moon_paper.npz')
    elif mode == 1:
        moondata(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5])
        data = np.load('moon_randoms.npz')
    else:
        data = np.load('moon_randoms_good.npz')

    x_s = torch.from_numpy(data['x_s']).float()
    y_s = torch.from_numpy(data['y_s']).float()
    x_t = torch.from_numpy(data['x_t']).float()

    return x_s, y_s, x_t

def grid_print(mode,x):

    if mode == 0:

        bias = 0.5
        xtx  = 1.6
        ytx  = -0.9

    else:

        bias = 2
        xtx  = 8
        ytx  = -3.9

    x_min, x_max = x[:, 0].min() - bias, x[:, 0].max() + bias
    y_min, y_max = x[:, 1].min() - bias, x[:, 1].max() + bias
    x_gp , y_gp  = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    z            = torch.from_numpy(np.c_[x_gp.ravel(), y_gp.ravel()]).float()

    return x_gp, y_gp, xtx, ytx, z
