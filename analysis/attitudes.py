
#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from math_helpers import rotations as rot
from math_helpers import matrices as mat
from math_helpers import vectors, quaternions
from numpy.linalg import norm
import numpy as np


if __name__ == "__main__":

    #1 true observation vectors in inertial frame
    v1_N = [-0.1517, -0.9669, 0.2050]
    v2_N = [-0.8393, 0.4494, -0.3044]
    v1_N = v1_N / np.linalg.norm(v1_N)
    v1_N = v1_N / np.linalg.norm(v1_N)
    
    # set up random measured attitude states
    v1_B = [0.8273, 0.5541, -0.0920]
    v2_B = [-0.8285, 0.5522, -0.0955]
    v1_B = v1_B / np.linalg.norm(v1_B)
    v2_B = v2_B / np.linalg.norm(v2_B)
    
    # estimated observation vectors in body frame
    tset_B, BbarT = rot.triad(v1_B, v2_B)

    # estimated observation vectors in inertial frame
    tset_N, NBarT = rot.triad(v1_N, v2_N)

    # transforms estimated attitude from the inertial to body frame
    BbarN = mat.mxm(BbarT, mat.mtranspose(NBarT))
    print(BbarN)


    #2 testing error estimation
    BN_true = [[0.963592, 0.187303, 0.190809],
               [-0.223042, 0.956645, 0.187303],
               [-0.147454, -0.223042, 0.963592]]

    BbarN = [[0.969846, 0.17101, 0.173648],
             [-0.200706, 0.96461, 0.17101],
             [-0.138258, -0.200706, 0.969846]]

    # compare the estimated body frame attitude with true attitude
    # should be near identity
    BbarB = mat.mxm(BbarN, mat.mtranspose(BN_true))

    # determine the magnitude of angle error
    prv_e, prv_a = rot.prv_axis(BbarB)
    error = np.rad2deg(prv_a)
    print(error)
    

    # davenport q method
    angles = np.deg2rad([30, 20, -10])
    BN_true = [[0.813798, 0.469846, -0.34202],
               [-0.543838, 0.823173, -0.163176],
               [0.204874, 0.318796, 0.925417]]
    v1_N = [1, 0, 0]
    v2_N = [0, 0, 1]
    v1_B_true = mat.mxv(BN_true, v1_N)
    v2_B_true = mat.mxv(BN_true, v2_N)
    print(v1_B_true)
    print(v2_B_true)

    # set up random measured attitude states
    v1_B = [0.8190, -0.5282, 0.2242]
    v2_B = [-0.3138, -0.1584, 0.9362]
    v1_B = v1_B / np.linalg.norm(v1_B)
    v2_B = v2_B / np.linalg.norm(v2_B)

    # setup q-method parameters
    w1 = 1
    w2 = 1

    # compute using q-method
    daven_q = rot.davenportq([v1_N, v2_N], [v1_B, v2_B], [w1, w2], sensors=2)
    print(daven_q)

    # get the dcm
    BbarN = quaternions.quat2dcm(daven_q)
    print(BbarN)

    # error
    BbarB = mat.mxm(BbarN, mat.mtranspose(BN_true))
    prv_e, prv_a = rot.prv_axis(BbarB)
    print(np.rad2deg(prv_a))

    # daven-q example 2
    v1B = [0.8273, 0.5541, -0.0920]
    v2B = [-0.8285, 0.5522, -0.0955]
    v1N = [-0.1517, -0.9669, 0.2050]
    v2N = [-0.8393, 0.4494, -0.3044]
    v1B = v1B / np.linalg.norm(v1B)
    v2B = v2B / np.linalg.norm(v2B)
    v1N = v1N / np.linalg.norm(v1N)
    v2N = v2N / np.linalg.norm(v2N)
    w1, w2 = 2, 1
    daven_q = rot.davenportq([v1N, v2N], [v1B, v2B], [w1, w2], sensors=2)
    BbarN = quaternions.quat2dcm(daven_q)
    print(BbarN)