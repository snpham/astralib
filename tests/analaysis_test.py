#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import rotations, vectors, quaternions, matrices
from math_helpers import mass_properties as mp
import numpy as np


def test_inertia():
    """tests cg_tensor and paxis (parallel axis) functions; also test
    mrp2dcm, mxadd, T_inertiaframe functions
    """
    # first test
    mass = 12.5
    I_cg = [[10., 1., -1.], [1., 5., 1], [-1., 1., 8.]]
    # spacecraft cg position, inertial
    rvec_N = [-0.5, 0.5, 0.25]
    # transform from inertial to body
    euler = [-10, 10, 5]
    BN = rotations.euler2dcm(np.deg2rad(euler), sequence='321')
    # spacecraft cg position, body
    rvec_B = matrices.mxv(BN, rvec_N)
    # parallel axis theorem
    cg_tensor = mp.cg_tensor(rvec_B)
    paxis = matrices.mxscalar(mass, cg_tensor)

    # inertia at point P
    J_P = matrices.mxadd(I_cg, paxis)
    J_P_truth = [[12.32125207,  4.19755562, -0.15813867], 
                 [ 4.19755562,  9.86047157,  0.42847142], 
                 [-0.15813867,  0.42847142, 14.88077637]]
    assert np.allclose(J_P, J_P_truth)

    # test 2
    # inertia matrix frame transform from body to D using MRP's
    Ic_B = [[10, 1, -1], [1, 5, 1], [-1, 1, 8]]
    mrp = [0.1, 0.2, 0.3]
    T_DB = quaternions.mrp2dcm(mrp)

    # Inertia transform
    Tmatrix = matrices.mxadd(m2=np.eye(3), m1=T_DB)
    Ic_D = mp.T_inertiaframe(Imatrix=Ic_B, Tmatrix=Tmatrix)
    Ic_D_truth = [[ 5.42779505, -1.77341012,  1.37988231],
                  [-1.77341012,  9.27952214, -0.53047352],
                  [ 1.37988231, -0.53047352,  8.29268281]]
    assert np.allclose(Ic_D, Ic_D_truth)
    # get principle inertias (eigenvalues)
    eigvals, eigvecs = np.linalg.eig(a=Ic_B)
    eigvals_truth = [ 4.41312549, 10.47419366,  8.11268085]
    assert np.allclose(eigvals, eigvals_truth)


def test_get_kineticenergy():
    """test rotational_kenergy function
    """
    # computing rotational kinetic energy from inertia tensor
    # and angular rate
    Icg_B = [[10, 1, -1], [1, 5, 1], [-1, 1, 8]]
    w = [0.01, -0.01, 0.01]
    T_rot = mp.rotational_kenergy(Icg_B, w)
    T_rot_truth = 0.00085
    assert np.allclose(T_rot, T_rot_truth)







if __name__ == '__main__':
    pass
    