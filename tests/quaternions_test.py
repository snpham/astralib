import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from math_helpers import matrices as mat
from math_helpers import quaternions as quat



def test_quat_add_scalar():
    """tests qxscalar, qadd
    """
    assert quat.qxscalar(scalar=1, quat=[1, 0, 0, 0]) == [1, 0, 0, 0]
    assert quat.qadd(quat1=[1, 0, 0, 0], quat2=[1, 0, 0, 0]) == [2, 0, 0, 0]


def test_qxq():
    """tests qxq and qxq2
    """
    # given two quat
    quat1 = [3, 1, -2, 1]
    quat2 = [2, -1, 2, 3]
    qset = quat.qxq(quat1, quat2)
    qset_truth = [8, -9, -2, 11]
    assert np.allclose(qset, qset_truth)
    qset = quat.qxq2(quat1, quat2)
    assert np.allclose(qset, qset_truth)


def test_qvq():
    """tests quaternion operators
    """
    # given a vector rotation quaternion and vector in a reference
    # frame
    qset = [np.sqrt(3)/2, 0, 0, 1/2]
    v1 = [1, 0, 0]
    # compute and test quaternion operator
    wvec = quat.q_operator_vector(qset, v1)
    wvec_truth = [1/2, np.sqrt(3)/2, 0]
    assert np.allclose(wvec, wvec_truth)
    # 2nd test - rotating over pi/3 - vector operator
    qset = [1/2, 1/2, 1/2, 1/2]
    wvec = quat.q_operator_vector(qset, v1)
    wvec_truth = [0, 1, 0]
    assert np.allclose(wvec, wvec_truth)    
    # 3rd test - using frame operator
    wvec = quat.q_operator_frame(qset, v1)
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
    qset = quat.dcm2quat_sheppard(T_dcm)
    qset_truth = (0.961798, -0.145650, 0.202665, 0.112505)
    assert np.allclose(qset, qset_truth)

    # quat2dcm basic
    qset = [0.235702, 0.471405, -0.471405, 0.707107]
    dcm = quat.quat2dcm(qset)
    dcm_true = [[-0.444444, -0.111112, 0.888889], 
                [-0.777778, -0.444444, -0.444445], 
                [0.444445, -0.888889, 0.111110]]
    assert np.allclose(dcm, dcm_true)

    # 2nd quat2dcm and sheppard test
    bn = [[-0.529403, -0.467056, 0.708231], 
          [-0.474115, -0.529403, -0.703525], 
          [0.703525, -0.708231, 0.0588291]]
    epset = quat.dcm2quat(bn)
    epset_true = [0.0024254, 0.4850696, -0.4850696, 0.7276048]
    assert np.allclose(epset, epset_true, rtol=0, atol=1e-02) # not accurate
    epset = quat.dcm2quat_sheppard(bn)
    assert np.allclose(epset, epset_true,rtol=0, atol=1e-06)

    # 3rd quat2dcm/sheppard test
    q_bn = [0.774597,0.258199,0.516398,0.258199]
    q_fb = [0.359211,0.898027,0.179605,0.179605]
    dcm_bn = quat.quat2dcm(q_bn)
    dcm_fb = quat.quat2dcm(q_fb)
    dcm_fn = mat.mxm(dcm_fb, dcm_bn)
    q_fn = quat.dcm2quat_sheppard(dcm_fn)
    q_fn = quat.dcm2quat(dcm_fn)
    q_fn_true = [0.0927449, -0.8347526, -0.5101265, 0.1855009]
    assert np.allclose(q_fn, q_fn_true)


def test_euler2quat():
    """tests euler2quat function
    """
    aset = np.deg2rad([20, 10, -10])
    qset = quat.euler2quat(aset[0], aset[1], aset[2], sequence='321')
    qset_true = [0.9760079, -0.1005818, 0.0704281, 0.1798098]
    assert np.allclose(qset, qset_true)


def test_crps():
    """tests crp2dcm and dcm2crp functions
    """
    # crp to dcm
    sigmas = [0.1, 0.2, 0.3]
    dcm = quat.crp2dcm(sigmas)
    dcm_true = [[ 0.77192982,  0.56140351, -0.29824561],
                [-0.49122807,  0.8245614,  0.28070175],
                [ 0.40350877, -0.07017544,  0.9122807 ]]
    assert np.allclose(dcm, dcm_true)

    # dcm to crp's
    dcm = [[0.333333,-0.666667,0.666667],
           [0.871795,0.487179,0.0512821],
           [-0.358974,0.564103,0.74359]]
    crp = quat.dcm2crp(dcm)
    crp_true = [-0.2,  -0.4, -0.6]
    assert np.allclose(crp, crp_true)



def test_mrps():
    """tests quatmrp, quat2mrps, mrp_shadow, dcm2mrp
    """
    # given euler parameters b
    qset = [0.961798, -0.145650, 0.202665, 0.112505]
    # compute and test quat to mrp and mrp shadowset
    sigma = quat.quat2mrp(qset=qset)
    sigma_sh = quat.quat2mrps(qset=qset)
    sigma_truth = (-0.0742431, 0.103306, 0.0573479)
    sigma_sh_truth = (3.81263, -5.30509, -2.945)
    assert np.allclose(sigma, sigma_truth)
    assert np.allclose(sigma_sh, sigma_sh_truth)

    # mrp shadow set
    sigma = [0.1, 0.2, 0.3]
    shadow = quat.mrp_shadow(sigma)
    shadow_true = [-0.7142857, -1.4285714, -2.1428571]
    assert np.allclose(shadow, shadow_true)

    # dcm to mrp
    dcm = [[0.763314, 0.0946746, -0.639053],
	       [-0.568047, -0.372781, -0.733728],
          [-0.307692, 0.923077, -0.230769]]
    mrp = quat.dcm2mrp(dcm)
    mrp_true = [-0.5,  0.1,  0.2]
    assert np.allclose(mrp, mrp_true)
