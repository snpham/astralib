#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import matrices as mat
from math_helpers import rotations, vectors, quaternions, matrices
import numpy as np

def test_mxm():
    """tests matrix transpose and multiplication
    """
    m1 =  mat.mtranspose([[0.0, 1.0, 0.0], [1, 0, 0], [0, 0, -1]])
    m2 = [[1/2, np.sqrt(3)/2, 0.], [0., 0., 1.], [np.sqrt(3)/2, -1/2., 0,]]
    m3 = mat.mxm(m2=m2, m1=m1)
    m3_truth = ([np.sqrt(3)/2, 1/2, 0], [0, 0, -1], [-1/2, np.sqrt(3)/2, 0])
    assert np.array_equal(m3, m3_truth)


def test_euler_rotation():
    """321 and 313; tests rotate_x, rotate_y, rotate_z, euler2dcm,
    euler2dcm2, dcm2euler, and euler2dcm functions
    """
    # euler angles for n2b, n2f
    b1, b2, b3 = np.deg2rad([30, -45, 60])
    f1, f2, f3 = np.deg2rad([10., 25., -15.])
    # compute rotation matrices
    T_n2b = rotations.euler2dcm(b1, b2, b3, '321')
    T_n2f = rotations.euler2dcm(f1, f2, f3, '321')
    T_f2b = mat.mxm(m2=T_n2b, m1=mat.mtranspose(T_n2f))
    T_n2b2 = rotations.euler2dcm(b1, b2, b3, '321')
    T_n2f2 = rotations.euler2dcm(f1, f2, f3, '321')
    # comparing euler2dcm and euler2dcm functions
    assert np.array_equal(T_n2b, T_n2b2)
    assert np.array_equal(T_n2f, T_n2f2)
    T_f2b_truth = ([ 0.303371,-0.0049418, 0.952859], 
                   [-0.935314, 0.1895340, 0.298769], 
                   [-0.182075,-0.9818620, 0.052877])
    # comparing rotation matrix product with truth
    assert np.allclose(T_f2b, T_f2b_truth)
    # computing euler angles from T matrix
    a1, a2, a3 = rotations.dcm2euler(dcm=T_f2b_truth, sequence='321')
    a1st, a2nd, a3rd = [np.rad2deg(angle) for angle in [a1, a2, a3]]
    # comparing computed angle with truth
    # FIXME 3rd angle is -79 deg in book
    assert np.allclose([a1st, a2nd, a3rd], ([-0.933242, -72.3373, 79.9635]))
    # get back matrix from angles
    T_f2b = rotations.euler2dcm(a1, a2, a3, '321')
    assert np.allclose(T_f2b, T_f2b_truth)

    # another dcm rotation test
    dcm = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    dcmx = rotations.rotate_x(np.deg2rad(90))
    dcmy = rotations.rotate_y(np.deg2rad(-90))
    dcmy = mat.mtranspose(dcmy)
    br = mat.mxm(dcmx, dcmy)
    br_true = [[0.0, 0.0, -1.0],
               [1.0, 0.0, 0.0], 
               [0.0, -1.0, 0.0]]
    assert np.allclose(br, br_true)

    # manual composite rotation
    sqrt = np.sqrt
    bn = [[1/3, 2/3, -2/3], [0, 1/sqrt(2), 1/sqrt(2)], [4/(3*sqrt(2)), -1/(3*sqrt(2)), 1/(3*sqrt(2))]]
    fn = [[3/4, -2/4, sqrt(3)/4],[-1/2, 0, sqrt(3)/2], [-sqrt(3)/4, -2*sqrt(3)/4, -1/4]]
    nf = mat.mtranspose(fn)
    bf = mat.mxm(bn, nf)
    bf_true = [[-0.37200847, -0.74401694, -0.55502117], 
               [-0.04736717, 0.61237244, -0.78914913], 
               [0.92701998, -0.26728038, -0.26304971]]
    assert np.allclose(bf, bf_true)

    # testing 313 sequence
    angles = np.deg2rad((10, 20, 30))
    dcm1 = rotations.euler2dcm(angles[0], angles[1], angles[2], sequence='321')
    eulers = np.rad2deg(rotations.dcm2euler(dcm1, sequence='313'))
    eulers_true = [40.64234205, 35.53134776, -36.05238873]
    assert np.allclose(eulers, eulers_true)

    # testing 313 sequence with composite rotation
    ang2 = np.deg2rad((-5, 5, 5))
    dcm_BN = rotations.euler2dcm2(angles[0], angles[1], angles[2], sequence='321')
    dcm_RN = rotations.euler2dcm2(ang2[0], ang2[1], ang2[2], sequence='321')
    dcm_NR = matrices.mtranspose(dcm_RN)
    dcm = matrices.mxm(dcm_BN, dcm_NR)
    eulers = np.rad2deg(rotations.dcm2euler(dcm, sequence='321'))
    eulers_true = [13.22381821, 16.36834338, 23.61762825]
    assert np.allclose(eulers, eulers_true)


def test_dcm_rates():
    """tests dcm_rate function
    """
    bn = [[-0.87097, 0.45161, 0.19355], 
          [-0.19355, -0.67742, 0.70968], 
          [0.45161, 0.58065, 0.67742]]
    wbn = [0.1, 0.2, 0.3]
    wbn_tl = matrices.skew_tilde(wbn)
    rates = rotations.dcm_rate(omega_tilde=wbn_tl, dcm=bn)
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
    phi = rotations.prv_angle(dcm=T_dcm)
    phi_deg = np.rad2deg(phi)
    assert np.allclose(phi_deg, 31.7762)
    # compute and test principal rotation axis
    evec, phi = rotations.prv_axis(dcm=T_dcm)
    phi_deg = np.rad2deg(phi)
    evec_truth =(-0.532035, 0.740302, 0.410964)
    assert np.allclose(phi_deg, 31.7762)
    assert np.allclose(evec, evec_truth)

    # second prv axis test
    dcm = [[0.925417, 0.336824, 0.173648],
           [0.0296956, -0.521281, 0.852869],
           [0.377786, -0.784102, -0.492404]]
    e, angle = rotations.prv_axis(dcm)
    e_true = [0.975550, 0.121655, 0.183032]
    angle_true = 2.146152
    assert np.allclose(e, e_true)
    assert np.allclose(angle, angle_true)

    # third prv axis test
    angles = np.deg2rad((20, -10, 120))
    e, angle = rotations.prv_axis(rotations.euler2dcm( \
            angles[0], angles[1], angles[2], sequence='321'))
    e_true = [0.975550, 0.121655, 0.183032]
    angle_true = 2.146152
    assert np.allclose(e, e_true)
    assert np.allclose(angle, angle_true)



def test_triad_method():
    """tests TRIAD algorithm, mxv, mtranspose, and vcrossv 
    functions
    """
    # given a dcm of Body relative to Nertial
    T_dcm = [[ 0.8138,  0.4698, -0.3420], 
             [-0.5438,  0.8232, -0.1632], 
             [ 0.2049,  0.3188,  0.9254]]
    v1 = [1, 0, 0] # first known inertial vector
    v2 = [0, 0, 1] # second known inertial vector
    # compute and test vectors in Body frame
    v1_B = matrices.mxv(m1=T_dcm, v1=v1)
    v2_B = matrices.mxv(m1=T_dcm, v1=v2)
    v1_B_truth = [0.8138, -0.5438, 0.2049]
    v2_B_truth = [-0.3420, -0.1632, 0.9254]
    assert np.allclose(v1_B, v1_B_truth)
    assert np.allclose(v2_B, v2_B_truth)
    # setting offset to simulate estimation
    v1_B = [0.8190, -0.5282, 0.2242]
    v2_B = [-0.3138, -0.1584, 0.9362]
    # compute and test TRIAD method function
    tset, Tmat = rotations.triad(v1=v1_B, v2=v2_B)
    tset = [np.round(t, 4) for t in tset]
    t1_truth = (0.8190, -0.5282, 0.2242)
    t2_truth = (-0.4593, -0.8376, -0.2957)
    t3_truth = (0.344, 0.1392, -0.9286)
    assert np.allclose(tset[0], t1_truth)
    assert np.allclose(tset[1], t2_truth)
    assert np.allclose(tset[2], t3_truth)


def test_davenportq_method():
    """tests davenport algorithm, mxscalar, vxv, mxadd, mxsub,
    mtranspose, and quat2dcm functions
    """
    # given inertial sensor measurements
    v1 = [1, 0, 0] # first known inertial vector
    v2 = [0, 0, 1] # second known inertial vector
    v1_B = [0.8190, -0.5282, 0.2242] # setting offset to simulate estimation
    v2_B = [-0.3138, -0.1584, 0.9362] # setting offset to simulate estimation
    vset_nrtl = [v1, v2]
    vset_body = [v1_B, v2_B]
    weights = [1, 1]
    # compute and test quaternion output
    qset = rotations.davenportq(vset_nrtl, vset_body, weights, sensors=2)
    qset_truth = [0.9481, -0.1172, 0.1414, 0.2597]
    assert np.allclose(qset, qset_truth, rtol=0, atol=1e-04)





if __name__ == "__main__":





    ## euler rotations
    angle = np.deg2rad(90)
    print(rotations.rotate_x(angle))
    print(rotations.rotate_y(angle))
    print(rotations.rotate_z(angle))

    # matrix multiplication
    matrix1 = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
    matrix2 = [[1/2, np.sqrt(3)/2, 0], [0, 0, 1], [np.sqrt(3)/2, -1/2, 0]]
    print(np.vstack(matrix1))
    print(np.vstack(matrix2))
    matrix1 = mat.mtranspose(m1=matrix1)
    print(mat.mxm(m2=matrix2, m1=matrix1))

    vector = [0, 1, 2]
    print(mat.skew_tilde(vector))

