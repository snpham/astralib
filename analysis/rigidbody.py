#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import matrices, quaternions, vectors, rotations
from math_helpers import mass_properties as mp
import numpy as np


if __name__ == "__main__":


    # inertia matrix frame transform from body to D using MRP's
    Ic_B = [[10, 1, -1], [1, 5, 1], [-1, 1, 8]]
    mrp = [0.1, 0.2, 0.3]
    T_DB = quaternions.mrp2dcm(mrp)
    Tmatrix = matrices.mxadd(m2=np.eye(3), m1=T_DB)
    Ic_D = mp.T_inertiaframe(Imatrix=Ic_B, Tmatrix=Tmatrix)
    # print(Ic_D)

    eigvals, eigvecs = np.linalg.eig(a=Ic_B)
    # print(eigvals)
    # print(eigvecs)

#########################

    # computing rotational kinetic energy from inertia tensor
    # and angular rate
    T_rot = mp.rotational_kenergy(Ic_B, [0.01, -0.01, 0.01])
    # print(T_rot)

#########################

    # kinetic energy
    # r = [[1, -1, 2], [-1, -3, 2], [2, -1, -1], [3, -1, -1]]
    # print(r[0])
    # rdot = [[2, 1, 1], [0, -1, 1], [3, 2, -1], [0, 0, 1]]
    # # te = 0.5*

#########################

    inertiacg = [[10., 1., -1.], [1., 5., 1], [-1., 1., 8.]]
    w = [0.01, -0.01, 0.01]
    v1 = matrices.mxv(m1=inertiacg, v1=w)
    v2 = vectors.vxvT(v1=v1, v2=w)
    trot = 0.5*v2
    print(trot)

#########################

    rvec_N = [-0.5, 0.5, 0.25]
    euler = [-10, 10, 5]
    BN = rotations.euler2dcm(np.deg2rad(euler), sequence='321')
    rvec_B = matrices.mxv(BN, rvec_N)
    cg_tensor = mp.cg_tensor(rvec_B)
    paxis = matrices.mxs(12.5, cg_tensor)
    J = matrices.mxadd(inertiacg, paxis)
    print(J)
