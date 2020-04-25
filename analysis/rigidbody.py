#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import matrices, quaternions, vectors
from math_helpers import mass_properties as mp
import numpy as np


if __name__ == "__main__":

    # testing angular momentum computation Hc = Iw
    # omega = [0.01, -0.01, 0.01]
    # inertia = [[10., 1., -1.], [1., 5., 1], [-1., 1., 8.]]
    # angmo = matrices.mxv(m1=inertia, v1=omega)
    # print(angmo)

    # # testing inertia matrix computation
    # inertiacg = [[10., 1., -1.], [1., 5., 1], [-1., 1., 8.]]
    # r = [-0.5, 0.5, 0.25]
    # cg_tensor = mp.cg_tensor(rvec=r)
    # # r_skew = matrices.skew_tilde(r)
    # # r_skew_t = matrices.mtranspose(r_skew)
    # # cg_tensor = matrices.mxm(r_skew, r_skew_t)
    # # cg_tensor = [[r[1]**2+r[2]**2, -r[0]*r[1], -r[0]*r[2]],
    # #              [-r[0]*r[1], r[0]**2+r[2]**2, -r[1]*r[2]],
    # #              [-r[0]*r[2], -r[1]*r[2], r[0]**2+r[1]**2]]
    # inertianew = np.zeros((3,3))
    # cg_scaled = matrices.mxscalar(scalar=12.5, m1=cg_tensor)
    # inertianew = matrices.mxadd(m2=inertiacg, m1=cg_scaled)
    # # inertianew[0][0] = inertiacg[0][0] + cg_scaled[0][0]
    # # inertianew[0][1] = inertiacg[0][1] + cg_scaled[0][1]
    # # inertianew[0][2] = inertiacg[0][2] + cg_scaled[0][2]
    # # inertianew[1][0] = inertiacg[1][0] + cg_scaled[1][0]
    # # inertianew[1][1] = inertiacg[1][1] + cg_scaled[1][1]
    # # inertianew[1][2] = inertiacg[1][2] + cg_scaled[1][2]
    # # inertianew[2][0] = inertiacg[2][0] + cg_scaled[2][0]
    # # inertianew[2][1] = inertiacg[2][1] + cg_scaled[2][1]
    # # inertianew[2][2] = inertiacg[2][2] + cg_scaled[2][2]
    # print(inertianew)

    # # inertia matrix frame transform
    # Ic = [[10, 1, -1], [1, 5, 1], [-1, 1, 8]]
    # sigset = [0.1, 0.2, 0.3]
    # I_dcm = quaternions.mrp2dcm(sigset)
    # # print(I_dcm)
    # Tmatrix = matrices.mxadd(m2=np.eye(3), m1=I_dcm)
    # inertia_newframe = mp.T_inertiaframe(Imatrix=Ic, Tmatrix=Tmatrix)
    # print(inertia_newframe)
    # eigvals, eigvecs = np.linalg.eig(a=Ic)
    # print(eigvals)
    # print(eigvecs)

    # kinetic energy
    # r = [[1, -1, 2], [-1, -3, 2], [2, -1, -1], [3, -1, -1]]
    # print(r[0])
    # rdot = [[2, 1, 1], [0, -1, 1], [3, 2, -1], [0, 0, 1]]
    # # te = 0.5*

    inertiacg = [[10., 1., -1.], [1., 5., 1], [-1., 1., 8.]]
    w = [0.01, -0.01, 0.01]
    v1 = matrices.mxv(m1=inertiacg, v1=w)
    v2 = vectors.vxv(v1=v1, v2=w)
    trot = 0.5*v2
    print(trot)

