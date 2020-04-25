#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import matrices, quaternions, rotations
import numpy as np


if __name__ == "__main__":
    # a = [[1, 0, 1], [2, 1, -1], [0, 1, 2]]
    # a_inv = matrices.mxscalar(scalar=1/5., m1=[[3, 1, -1], [-4, 2, 3], [2, -1, 1]])
    # iden = matrices.mxm(m2=a_inv, m1=a)
    # print(iden) 

    # # test retrieving dcm from prv
    # rvec = [1, 1, 1]
    # rvec = rvec/np.linalg.norm(rvec)
    # phi = 2/3*np.pi
    # dcm = quaternions.prv2dcm(e1=rvec[0], e2=rvec[1], e3=rvec[2], theta=phi)
    # print(np.vstack(dcm))
    # rvec2 = matrices.mxv(m1=dcm, v1=rvec)
    # print(rvec2)

    # testing rotation functions
    alpha = np.pi/6.
    beta = np.pi/3.
    rotz = rotations.rotate_z(alpha)
    roty = rotations.rotate_y(beta)
    T_rot = matrices.mxm(m2=roty, m1=rotz)
    print(T_rot)
    evec, theta = quaternions.prv_axis(dcm=T_rot)
    print(evec, np.rad2deg(theta))
    print([-1, 3.7321, 1.732]/np.linalg.norm([-1, 3.7321, 1.732]))
    v = [np.sqrt(3)/2, 1/2, -np.sqrt(3)]
    vout = matrices.mxv(m1=T_rot, v1=v)
    print(vout)