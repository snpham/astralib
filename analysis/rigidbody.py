#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers.matrices import *
from math_helpers.quaternions import *
from math_helpers.vectors import *
from math_helpers.mass_properties import *
import numpy as np


if __name__ == "__main__":

    # # testing angular momentum computation Hc = Iw
    # eulerangles = [-10, 10, 5]
    # rotation = '321'
    # omega = [0.01, -0.01, 0.01]
    # # omegaT = vtranspose(omega)
    # inertia = [[10., 1., -1.], [1., 5., 1], [-1., 1., 8.]]
    # angmo = mxv(m1=inertia, v1=omega)
    # print(angmo)

    # testing inertia matrix computation
    # inertiacg = [[10., 1., -1.], [1., 5., 1], [-1., 1., 8.]]
    # r = [-0.5, 0.5, 0.25]
    # r_skew = skew_tilde(r)
    # r_skew_t = mtranspose(r_skew)
    # cg_tensor = mxm(r_skew, r_skew_t)
    # # cg_tensor = [[r[1]**2+r[2]**2, -r[0]*r[1], -r[0]*r[2]],
    # #              [-r[0]*r[1], r[0]**2+r[2]**2, -r[1]*r[2]],
    # #              [-r[0]*r[2], -r[1]*r[2], r[0]**2+r[1]**2]]
    # inertianew = np.zeros((3,3))
    # inertianew[0][0] = inertiacg[0][0] + 12.5*cg_tensor[0][0]
    # inertianew[0][1] = inertiacg[0][1] + 12.5*cg_tensor[0][1]
    # inertianew[0][2] = inertiacg[0][2] + 12.5*cg_tensor[0][2]
    # inertianew[1][0] = inertiacg[1][0] + 12.5*cg_tensor[1][0]
    # inertianew[1][1] = inertiacg[1][1] + 12.5*cg_tensor[1][1]
    # inertianew[1][2] = inertiacg[1][2] + 12.5*cg_tensor[1][2]
    # inertianew[2][0] = inertiacg[2][0] + 12.5*cg_tensor[2][0]
    # inertianew[2][1] = inertiacg[2][1] + 12.5*cg_tensor[2][1]
    # inertianew[2][2] = inertiacg[2][2] + 12.5*cg_tensor[2][2]
    # print(inertianew)

    # inertia matrix frame transform
    Ic = [[10, 1, -1], [1, 5, 1], [-1, 1, 8]]
    sigset = [0.1, 0.2, 0.3]
    I_dcm = mrp2dcm(sigset)
    # print(I_dcm)
    Tmatrix = mxadd(m2=np.eye(3), m1=I_dcm)
    inertia_newframe = T_inertiaframe(Imatrix=Ic, Tmatrix=Tmatrix)
    print(inertia_newframe)


