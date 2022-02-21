#!/usr/bin/env python3
import sys 
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import matrices as mat
from math_helpers import rotations as rot
from math_helpers import quaternions as quat


def test_mxm():
    """tests matrix transpose and multiplication
    """
    m1 =  mat.mT([[0.0, 1.0, 0.0], [1, 0, 0], [0, 0, -1]])
    m2 = [[1/2, np.sqrt(3)/2, 0.], [0., 0., 1.], [np.sqrt(3)/2, -1/2., 0,]]
    m3 = mat.mxm(m2=m2, m1=m1)
    m3_truth = ([np.sqrt(3)/2, 1/2, 0], [0, 0, -1], [-1/2, np.sqrt(3)/2, 0])
    assert np.array_equal(m3, m3_truth)


def test_euler_rotation():
    """321 and 313; tests rotate, euler2dcm,
    euler2dcm2, dcm2euler, and euler2dcm functions
    """
    # euler angles for n2b, n2f
    b = np.deg2rad([30, -45, 60])
    f = np.deg2rad([10., 25., -15.])
    # compute rotation matrices
    T_n2b = rot.euler2dcm(b, '321')
    T_n2f = rot.euler2dcm(f, '321')
    T_f2b = mat.mxm(m2=T_n2b, m1=mat.mT(T_n2f))
    T_n2b2 = rot.euler2dcm(b, '321')
    T_n2f2 = rot.euler2dcm(f, '321')
    # comparing euler2dcm and euler2dcm functions
    assert np.array_equal(T_n2b, T_n2b2)
    assert np.array_equal(T_n2f, T_n2f2)
    T_f2b_truth = ([ 0.303371,-0.0049418, 0.952859], 
                   [-0.935314, 0.1895340, 0.298769], 
                   [-0.182075,-0.9818620, 0.052877])
    # comparing rotation matrix product with truth
    assert np.allclose(T_f2b, T_f2b_truth)
    # computing euler angles from T matrix
    a1 = rot.dcm2euler(dcm=T_f2b_truth, sequence='321')
    a1st = [np.rad2deg(angle) for angle in a1]
    # comparing computed angle with truth
    # FIXME 3rd angle is -79 deg in book
    assert np.allclose(a1st, ([-0.933242, -72.3373, 79.9635]))
    # get back matrix from angles
    T_f2b = rot.euler2dcm(a1, '321')
    assert np.allclose(T_f2b, T_f2b_truth)

    # another dcm rotation test
    dcm = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    dcmx = rot.rotate(np.deg2rad(90), axis='x')
    dcmy = rot.rotate(np.deg2rad(-90), axis='y')
    dcmy = mat.mT(dcmy)
    br = mat.mxm(dcmx, dcmy)
    br_true = [[0.0, 0.0, -1.0],
               [1.0, 0.0, 0.0], 
               [0.0, -1.0, 0.0]]
    assert np.allclose(br, br_true)

    # manual composite rotation
    sqrt = np.sqrt
    bn = [[1/3, 2/3, -2/3], [0, 1/sqrt(2), 1/sqrt(2)], [4/(3*sqrt(2)), -1/(3*sqrt(2)), 1/(3*sqrt(2))]]
    fn = [[3/4, -2/4, sqrt(3)/4],[-1/2, 0, sqrt(3)/2], [-sqrt(3)/4, -2*sqrt(3)/4, -1/4]]
    nf = mat.mT(fn)
    bf = mat.mxm(bn, nf)
    bf_true = [[-0.37200847, -0.74401694, -0.55502117], 
               [-0.04736717, 0.61237244, -0.78914913], 
               [0.92701998, -0.26728038, -0.26304971]]
    assert np.allclose(bf, bf_true)

    # testing 313 sequence
    angles = np.deg2rad((10, 20, 30))
    dcm1 = rot.euler2dcm(angles, sequence='321')
    eulers = np.rad2deg(rot.dcm2euler(dcm1, sequence='313'))
    eulers_true = [40.64234205, 35.53134776, -36.05238873]
    assert np.allclose(eulers, eulers_true)

    # testing 313 sequence with composite rotation
    ang2 = np.deg2rad((-5, 5, 5))
    dcm_BN = rot.euler2dcm2(angles, sequence='321')
    dcm_RN = rot.euler2dcm2(ang2, sequence='321')
    dcm_NR = mat.mT(dcm_RN)
    dcm = mat.mxm(dcm_BN, dcm_NR)
    eulers = np.rad2deg(rot.dcm2euler(dcm, sequence='321'))
    eulers_true = [13.22381821, 16.36834338, 23.61762825]
    assert np.allclose(eulers, eulers_true)


def test_dcm_rates():
    """tests dcm_rate function
    """
    bn = [[-0.87097, 0.45161, 0.19355], 
          [-0.19355, -0.67742, 0.70968], 
          [0.45161, 0.58065, 0.67742]]
    wbn = [0.1, 0.2, 0.3]
    wbn_tl = mat.skew(wbn)
    rates = rot.dcm_rate(omega_tilde=wbn_tl, dcm=bn)
    rates_true = [[-0.148387, -0.319356,  0.07742], 
                  [ 0.306452, -0.077418,  0.009677], 
                  [-0.154839,  0.158064, -0.032258]]
    assert np.allclose(rates, rates_true)



def test_prv():
    """tests principal rotation axis and angle 
    (prv_angle, prv_axis) functions
    """
    # given a dcm
    T_dcm = [[ 0.892539,  0.157379, -0.422618], 
             [-0.275451,  0.932257, -0.234570], 
             [ 0.357073,  0.325773,  0.875426]]
    # compute and test angle of rotation
    phi = rot.prv_angle(dcm=T_dcm)
    phi_deg = np.rad2deg(phi)
    assert np.allclose(phi_deg, 31.7762)
    # compute and test principal rotation axis
    evec, phi = rot.prv_axis(dcm=T_dcm)
    phi_deg = np.rad2deg(phi)
    evec_truth =(-0.532035, 0.740302, 0.410964)
    assert np.allclose(phi_deg, 31.7762)
    assert np.allclose(evec, evec_truth)

    # second prv axis test
    dcm = [[0.925417, 0.336824, 0.173648],
           [0.0296956, -0.521281, 0.852869],
           [0.377786, -0.784102, -0.492404]]
    e, angle = rot.prv_axis(dcm)
    e_true = [0.975550, 0.121655, 0.183032]
    angle_true = 2.146152
    assert np.allclose(e, e_true)
    assert np.allclose(angle, angle_true)

    # third prv axis test
    angles = np.deg2rad((20, -10, 120))
    e, angle = rot.prv_axis(rot.euler2dcm( \
            angles, sequence='321'))
    e_true = [0.975550, 0.121655, 0.183032]
    angle_true = 2.146152
    assert np.allclose(e, e_true)
    assert np.allclose(angle, angle_true)



def test_triad_method():
    """tests TRIAD algorithm, mxv, mT, and vcrossv 
    functions
    """
    # truth values
    BNangles_true = np.deg2rad((30, 20, -10))
    BN_true = rot.euler2dcm(BNangles_true, sequence='321')
    BN_truth = [[ 0.81379768,  0.46984631, -0.34202014],
               [-0.54383814,  0.82317294, -0.16317591],
               [ 0.20487413,  0.31879578,  0.92541658]]
    assert np.allclose(BN_true, BN_truth)
    # true observation vectors in inertial frame
    v1_N = [1, 0, 0] # first known inertial vector
    v2_N = [0, 0, 1] # second known inertial vector
    # compute and test vectors in Body frame
    # true observation vectors in body frame
    v1_B_true = mat.mxv(BN_true, v1_N)
    v2_B_true = mat.mxv(BN_true, v2_N)
    v1_B_truth = [0.81379768, -0.54383814,  0.20487413]
    v2_B_truth = [-0.34202014, -0.16317591,  0.92541658]
    assert np.allclose(v1_B_true, v1_B_truth)
    assert np.allclose(v2_B_true, v2_B_truth)
    # set up random measured attitude states
    v1_B = [0.8190, -0.5282, 0.2242]
    v2_B = [-0.3138, -0.1584, 0.9362]
    v1_B = v1_B / np.linalg.norm(v1_B)
    v2_B = v2_B / np.linalg.norm(v2_B)
    # compute and test TRIAD method function
    # estimated observation vectors in body frame
    tset_B, BbarT = rot.triad(v1_B, v2_B)
    t1_truth = (0.81899104, -0.52819422,  0.22419755)
    t2_truth = (-0.45928237, -0.83763943, -0.29566855)
    t3_truth = (0.34396712,  0.13917991, -0.92860948)
    assert np.allclose(tset_B[0], t1_truth)
    assert np.allclose(tset_B[1], t2_truth)
    assert np.allclose(tset_B[2], t3_truth)
    # estimated observation vectors in inertial frame
    tset_N, NBarT = rot.triad(v1_N, v2_N)
    t1_truth = (1, 0, 0)
    t2_truth = (0, -1, 0)
    t3_truth = (0, 0, -1)
    assert np.allclose(tset_N[0], t1_truth)
    assert np.allclose(tset_N[1], t2_truth)
    assert np.allclose(tset_N[2], t3_truth)
    # transforms estimated attitude from the inertial to body frame
    BbarN = mat.mxm(BbarT, mat.mT(NBarT))
    BbarN_truth = [[ 0.81899104,  0.45928237, -0.34396712],
                   [-0.52819422,  0.83763943, -0.13917991],
                   [ 0.22419755,  0.29566855,  0.92860948]]
    assert np.allclose(BbarN, BbarN_truth)
    # compare the estimated body frame attitude with true attitude
    # should be near identity
    BbarB = mat.mxm(BbarN, mat.mT(BN_true))
    BbarB_truth = [[ 0.99992882, -0.0112026,  -0.00410552],
                   [ 0.0113209,   0.99948509,  0.03002319],
                   [ 0.00376707, -0.03006753,  0.99954077]]
    assert np.allclose(BbarB, BbarB_truth, rtol=0, atol=1e-04)
    # determine the magnitude of angle error
    prv_e, prv_a = rot.prv_axis(BbarB)
    error = np.rad2deg(prv_a)
    assert np.allclose(error, 1.8525322)    

    # test 2 of triad method
    # true observation vectors in inertial frame
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
    BbarN = mat.mxm(BbarT, mat.mT(NBarT))
    BbarN_true = [[ 0.41555875, -0.85509088,  0.31004921],
                  [-0.83393237, -0.49427603, -0.24545471],
                  [ 0.36313597, -0.15655922, -0.91848869]]
    assert np.allclose(BbarN, BbarN_true)

    # test 3 - testing error estimation
    BN_true = [[0.963592, 0.187303, 0.190809],
               [-0.223042, 0.956645, 0.187303],
               [-0.147454, -0.223042, 0.963592]]
    BbarN = [[0.969846, 0.17101, 0.173648],
             [-0.200706, 0.96461, 0.17101],
             [-0.138258, -0.200706, 0.969846]]
    # compare the estimated body frame attitude with true attitude
    # should be near identity
    BbarB = mat.mxm(BbarN, mat.mT(BN_true))
    # determine the magnitude of angle error
    prv_e, prv_a = rot.prv_axis(BbarB)
    error = np.rad2deg(prv_a)
    error_true = 1.8349476
    assert np.allclose(error, error_true)


def test_davenportq_method():
    """tests davenport algorithm, mxs, vxvT, mxadd, mxsub,
    mT, and quat2dcm functions
    """
    # given inertial sensor measurements
    v1_N = [1, 0, 0] # first known inertial vector
    v2_N = [0, 0, 1] # second known inertial vector
    v1_B = [0.8190, -0.5282, 0.2242] # setting offset to simulate estimation
    v2_B = [-0.3138, -0.1584, 0.9362] # setting offset to simulate estimation
    v1_B = v1_B / np.linalg.norm(v1_B)
    v2_B = v2_B / np.linalg.norm(v2_B)
    vset_nrtl = [v1_N, v2_N]
    vset_body = [v1_B, v2_B]
    weights = [1, 1]
    # compute and test quaternion output
    qset = rot.davenportq(vset_nrtl, vset_body, weights, sensors=2)
    qset_truth = [0.94806851, -0.11720728,  0.14137123,  0.2596974]
    try:
        assert np.allclose(qset, qset_truth)
    except AssertionError:
        assert np.allclose(-qset, qset_truth)
    # truth values
    angles = np.deg2rad([30, 20, -10])
    BN_true = [[0.813798, 0.469846, -0.34202],
               [-0.543838, 0.823173, -0.163176],
               [0.204874, 0.318796, 0.925417]]
    v1_B_true = mat.mxv(BN_true, v1_N)
    v2_B_true = mat.mxv(BN_true, v2_N)
    v1_B_truth = [0.813798, -0.543838,  0.204874]
    v2_B_truth = [-0.34202,  -0.163176,  0.925417]
    assert np.allclose(v1_B_true, v1_B_truth)
    assert np.allclose(v2_B_true, v2_B_truth)
    # get the dcm
    BbarN = quat.quat2dcm(qset)
    BbarN_true = [[ 0.8251428, 0.4592823,  -0.3289360],
                  [-0.5255613, 0.8376394, -0.1488135],
                  [ 0.2071823, 0.2956685, 0.9325532]]    
    assert np.allclose(BbarN, BbarN_true)
    # error
    BbarB = mat.mxm(BbarN, mat.mT(BN_true))
    prv_e, prv_a = rot.prv_axis(BbarB)
    prv_a = np.rad2deg(prv_a)
    prv_a_truth = 1.6954989
    assert np.allclose(prv_a, prv_a_truth)

    # daven-q test 2
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
    BbarN = quat.quat2dcm(daven_q)
    BbarN_true = [[ 0.4158105, -0.8549593,  0.3100744],
                  [-0.8338152, -0.4945165, -0.2453681],
                  [ 0.3631167, -0.1565181, -0.9185033]]
    assert np.allclose(BbarN, BbarN_true)


def test_quest_method():
    # quest using example 1
    BN_true = [[ 0.813798, 0.469846, -0.34202],
               [-0.543838, 0.823173, -0.163176],
               [ 0.204874, 0.318796,  0.925417]]
    v1_N = [1, 0, 0]
    v2_N = [0, 0, 1]
    # set up random measured attitude states
    v1_B = [0.8190, -0.5282, 0.2242]
    v2_B = [-0.3138, -0.1584, 0.9362]
    v1_B = v1_B / np.linalg.norm(v1_B)
    v2_B = v2_B / np.linalg.norm(v2_B)

    w1, w2 = 1, 1
    # setting up davenport function to bypass eigval/eigvec computation
    # and use quest method
    crp = rot.davenportq([v1_N, v2_N], [v1_B, v2_B], [w1, w2], sensors=2, quest=True)
    crp_truth = [-0.1236021, 0.1491002, 0.2738740]
    assert np.allclose(crp, crp_truth)
    BbarN = quat.crp2dcm(crp)
    BbarN_truth = [[ 0.8251927, 0.4592204, -0.3288972],
                   [-0.5254815, 0.8376930, -0.1487934],
                   [ 0.2071859, 0.2956127,  0.9325701]]
    assert np.allclose(BbarN, BbarN_truth)
    BbarB = mat.mxm(BbarN, mat.mT(BN_true))
    BbarB_truth = [[ 0.9997925, -0.0170851,  0.0110910],
                   [ 0.0168412,  0.9996226,  0.0216997],
                   [-0.0114576, -0.0215082,  0.9997034]]
    assert np.allclose(BbarB, BbarB_truth)
    axis, ang = rot.prv_axis(BbarB)
    mag = np.rad2deg(np.linalg.norm(ang))
    mag_truth = 1.7009912
    assert np.allclose(mag, mag_truth)

    # test 2
        # quest example 2
    w1, w2 = 2, 1
    v1_B = [0.8273, 0.5541, -0.0920]
    v2_B = [-0.8285, 0.5522, -0.0955]
    v1_B = v1_B / np.linalg.norm(v1_B)
    v2_B = v2_B / np.linalg.norm(v2_B)
    v1_N = [-0.1517, -0.9669, 0.2050]
    v2_N = [-0.8393, 0.4494, -0.3044]
    v1_N = v1_N / np.linalg.norm(v1_N)
    v2_N = v2_N / np.linalg.norm(v2_N)
    crp = rot.davenportq([v1_N, v2_N], [v1_B, v2_B], [w1, w2], sensors=2, quest=True)
    crp_truth = [-31.83404638,  19.00448853,  -7.57577844]
    assert np.allclose(crp, crp_truth)
    BbarN = quat.crp2dcm(crp)
    BbarN_truth = [[ 0.41581032, -0.85495964,  0.31007386],
                   [-0.83381256, -0.49451739, -0.24537555],
                   [ 0.36312311, -0.15651379, -0.91850152]]
    assert np.allclose(BbarN, BbarN_truth)    
    qset_davenport = rot.davenportq([v1_N, v2_N], [v1_B, v2_B], [w1, w2], sensors=2, quest=False)
    BbarN_davenport = quat.quat2dcm(qset_davenport)
    assert np.allclose(BbarN, BbarN_davenport, rtol=0, atol=1e-05)



if __name__ == "__main__":

    pass