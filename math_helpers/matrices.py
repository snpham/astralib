#!/usr/bin/env python3
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import rotations


def mtranspose(m1):
    """computes the transpose of a square matrix
    """
    mt = np.zeros((len(m1),len(m1)))
    for ii in range(len(m1)):
        for jj in range(len(m1)):
            if ii == jj:
                mt[ii][jj] = m1[ii][jj]
            if ii != jj:
                mt[ii][jj] = m1[jj][ii]
    return mt


def mxadd(m2, m1):
    """matrix addition
    """
    m_out = np.zeros((len(m2),len(m1)))
    for ii in range(len(m2)):
        for jj in range(len(m1)):
            m_out[ii][jj] = m2[ii][jj]+m1[ii][jj]
    return m_out


def mxsub(m2, m1):
    """matrix subtraction
    """
    m_out = np.zeros((len(m2),len(m1)))
    for ii in range(len(m2)):
        for jj in range(len(m1)):
            m_out[ii][jj] = m2[ii][jj]-m1[ii][jj]
    return m_out


def mxm(m2, m1):
    """matrix multiplication; currently for 3x3 matrices
    :param m2: left matrix to be multiplied
    :param m1: right matrix to be multiplied
    :return m_out: matrix product m_out = [m2]*[m1]
    """
    m_out = np.zeros((len(m2),len(m1)))
    if len(m2) == 3 and len(m1) ==3:
        for ii in range(len(m2)):
            for jj in range(len(m1)):
                m_out[ii][jj] = m2[ii][0]*m1[0][jj] + m2[ii][1]*m1[1][jj] + m2[ii][2]*m1[2][jj]
    return m_out


def mxv(m1, v1):
    """multiplies vector by a matrix; currently for 3x3 and 4x4 matrices
    :param m1: nxn matrix to be multiplied
    :param v1: vector of n rows to be multiplied
    :return vout: resultant vector 
    """
    v_out = np.zeros(len(v1))
    if len(v1) == 4:
        for ii in range(len(v1)):
            v_out[ii] = m1[ii][0]*v1[0] + m1[ii][1]*v1[1] + m1[ii][2]*v1[2] + m1[ii][3]*v1[3]
        return v_out    
    if len(v1) == 3:
        for ii in range(len(v1)):
            v_out[ii] = m1[ii][0]*v1[0] + m1[ii][1]*v1[1] + m1[ii][2]*v1[2]
        return v_out     


def mxscalar(scalar, m1):
    """scales a matrix by a scalar
    """
    m_out = np.zeros((len(m1),len(m1)))
    for ii in range(len(m1)):
        for jj in range(len(m1)):
            m_out[ii][jj] = scalar * m1[ii][jj]
    return m_out


def skew_tilde(v1):
    """generates a skewed cross-product matrix from a vector
    """
    v_tilde = np.zeros((len(v1), len(v1)))
    v_tilde[0][1] = -v1[2]
    v_tilde[0][2] = v1[1]
    v_tilde[1][0] = v1[2]
    v_tilde[1][2] = -v1[0]
    v_tilde[2][0] = -v1[1]
    v_tilde[2][1] = v1[0]
    return v_tilde

    
if __name__ == '__main__':
    b1, b2, b3 = np.deg2rad([30, -45, 60])

    brotate = rotations.rotate_euler(b1, b2, b3, '321')
        #print(f'BN: {brotate}')

    f1, f2, f3 = np.deg2rad([10., 25., -15.])
    frotate = rotations.rotate_euler(f1, f2, f3, '321')
    frot = rotations.rotate_sequence(f1, f2, f3, '321')
    print(f'FN: {frotate}')
    if np.array_equal(frotate, frot):
        print("is equal")
    ftranspose = mtranspose(frotate) 
    # print(rotations.rotate_sequence(a1, a2, a3, '313'))

    matrix = mxm(brotate, ftranspose)
    print(f'result={matrix}')
    a1, a2, a3 = np.rad2deg(rotations.dcm_inverse(matrix, '321'))
    print(a1, a2, a3)
