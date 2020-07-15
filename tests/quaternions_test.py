import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from math_helpers import matrices, rotations, vectors, quaternions




def test_quat_add_scalar():
    """tests qxscalar, qadd
    """
    assert quaternions.qxscalar(scalar=1, quat=[1, 0, 0, 0]) == [1, 0, 0, 0]
    assert quaternions.qadd(quat1=[1, 0, 0, 0], quat2=[1, 0, 0, 0]) == [2, 0, 0, 0]


def test_qxq():
    """tests qxq and qxq2
    """
    # given two quaternions
    quat1 = [3, 1, -2, 1]
    quat2 = [2, -1, 2, 3]
    qset = quaternions.qxq(quat1, quat2)
    qset_truth = [8, -9, -2, 11]
    assert np.allclose(qset, qset_truth)
    qset = quaternions.qxq2(quat1, quat2)
    assert np.allclose(qset, qset_truth)


def test_qvq():
    """tests quaternion operators
    """
    # given a vector rotation quaternion and vector in a reference
    # frame
    quat = [np.sqrt(3)/2, 0, 0, 1/2]
    v1 = [1, 0, 0]
    # compute and test quaternion operator
    wvec = quaternions.q_operator_vector(quat, v1)
    wvec_truth = [1/2, np.sqrt(3)/2, 0]
    assert np.allclose(wvec, wvec_truth)
    # 2nd test - rotating over pi/3 - vector operator
    quat = [1/2, 1/2, 1/2, 1/2]
    wvec = quaternions.q_operator_vector(quat, v1)
    wvec_truth = [0, 1, 0]
    assert np.allclose(wvec, wvec_truth)    
    # 3rd test - using frame operator
    wvec = quaternions.q_operator_frame(quat, v1)
    wvec_truth = [0, 0, 1]
    assert np.allclose(wvec, wvec_truth)    


def test_quat2dcm():
    """tests quat2dcm and sheppards method
    """
    # sheppard
    # given a dcm
    T_dcm = [[ 0.892539,  0.157379, -0.422618], 
             [-0.275451,  0.932257, -0.234570], 
             [ 0.357073,  0.325773,  0.875426]]
    # compute and test dcm to quat using Sheppard
    quat = quaternions.dcm2quat_sheppard(T_dcm)
    quat_truth = (0.961798, -0.145650, 0.202665, 0.112505)
    assert np.allclose(quat, quat_truth)

    # quat2dcm basic
    qset = [0.235702, 0.471405, -0.471405, 0.707107]
    dcm = quaternions.quat2dcm(qset)
    dcm_true = [[-0.444444, -0.111112, 0.888889], 
                [-0.777778, -0.444444, -0.444445], 
                [0.444445, -0.888889, 0.111110]]
    assert np.allclose(dcm, dcm_true)

    # 2nd quat2dcm and sheppard test
    bn = [[-0.529403, -0.467056, 0.708231], 
          [-0.474115, -0.529403, -0.703525], 
          [0.703525, -0.708231, 0.0588291]]
    epset = quaternions.dcm2quat(bn)
    epset_true = [0.0024254, 0.4850696, -0.4850696, 0.7276048]
    assert np.allclose(epset, epset_true, rtol=0, atol=1e-02) # not accurate
    epset = quaternions.dcm2quat_sheppard(bn)
    assert np.allclose(epset, epset_true,rtol=0, atol=1e-06)

    # 3rd quat2dcm/sheppard test
    q_bn = [0.774597,0.258199,0.516398,0.258199]
    q_fb = [0.359211,0.898027,0.179605,0.179605]
    dcm_bn = quaternions.quat2dcm(q_bn)
    dcm_fb = quaternions.quat2dcm(q_fb)
    dcm_fn = matrices.mxm(dcm_fb, dcm_bn)
    q_fn = quaternions.dcm2quat_sheppard(dcm_fn)
    q_fn = quaternions.dcm2quat(dcm_fn)
    q_fn_true = [0.0927449, -0.8347526, -0.5101265, 0.1855009]
    assert np.allclose(q_fn, q_fn_true)


def test_euler2quat():
    """tests euler2quat function
    """
    aset = np.deg2rad([20, 10, -10])
    quat = quaternions.euler2quat(aset[0], aset[1], aset[2], sequence='321')
    quat_true = [0.9760079, -0.1005818, 0.0704281, 0.1798098]
    assert np.allclose(quat, quat_true)


def test_mrps():
    # given euler parameters b
    qset = [0.961798, -0.145650, 0.202665, 0.112505]
    # compute and test quat to mrp and mrp shadowset
    sigma = quaternions.quat2mrp(qset=qset)
    sigma_sh = quaternions.quat2mrps(qset=qset)
    sigma_truth = (-0.0742431, 0.103306, 0.0573479)
    sigma_sh_truth = (3.81263, -5.30509, -2.945)
    assert np.allclose(sigma, sigma_truth)
    assert np.allclose(sigma_sh, sigma_sh_truth)
