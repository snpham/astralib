#!/usr/bin/env python3
import matrices
from numpy.linalg import norm
import numpy as np
from pprint import pprint as pp


def triad(v1, v2):
    """TRIAD attitude estimation method. Uses two vector observations, 
    v1 and v2, to establish a frame tset with v1 being primary and
    v2 secondary.
    """
    t1 = v1
    v1xv2 = matrices.vxv(v1, v2)
    t2 = v1xv2 / norm(v1xv2)
    t3 = matrices.vxv(t1, t2)
    tset = [t1, t2, t3]
    # transform matrix for new tset frame to v frame
    t2vmatrix = matrices.mtranspose(tset)
    return tset, t2vmatrix


def davenportq(vset, n):
    pass



if __name__ == "__main__":
    #testing triad method
    v1 = [1, 0, 0]
    v2 = [0, 0, 1]
    tveci, t2i = triad(v1, v2)
    # print(tveci)
    # print(t2i)
    bn_actual = matrices.rotate_euler(a1=np.deg2rad(30), a2=np.deg2rad(20), a3=np.deg2rad(-10), 
                               sequence='321')
    v1out_a = matrices.mxv(bn_actual, v1)
    v2out_a = matrices.mxv(bn_actual, v2)
    v1out = [0.8190, -0.5282, 0.2242]
    v2out = [-0.3138, -0.1584, 0.9362]
    # print(v1out, v2out)
    tvec, tmatrix = triad(v1out, v2out)
    # print(tvec)
    # print(tmatrix)
    bn = matrices.matrix_multT(m2=tmatrix, m1=t2i)
    print(bn)
    bn_error = matrices.matrix_multT(m2=bn, m1=bn_actual)
    print(bn_error)