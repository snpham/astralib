#!/usr/bin/env python3
import numpy as np


def vtranspose(v1):
    if len(v1) > 1:
        vt = np.zeros(len(v1))
        print(vt)
        for i in range(len(v1[0, :])):
            vt[0] = v1[0, i]
    return vt


def vxm(v1, m1):
    """multiplies vector by a matrix; currently for length 3 vectors
    """
    v_out = np.zeros(len(v1)) 
    if len(v1) == 3:
        for ii in range(len(v1)):
            for jj in range(len(m1)):
                v_out[ii] = v1[0]*m1[0][ii] + v1[1]*m1[1][ii] + v1[2]*m1[2][ii]
        return v_out   


def vxv(v1, v2):
    """multiplies two vectors
    """
    vout = np.zeros(len(v1))
    print(v1, v2)
    vout[0] = v1[1]*v2[2] - v1[2]*v2[1]
    vout[1] = -(v1[0]*v2[2] - v1[2]*v2[0])
    vout[2] = v1[0]*v2[1] - v1[1]*v2[0]
    return vout


if __name__ == "__main__":
    pass