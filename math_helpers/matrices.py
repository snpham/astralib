#!/usr/bin/env python3
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def mT(m1):
    """computes the transpose of a square matrix
    :param m1: rotation matrix
    :return mt: m1 transpose
    """
    mt = np.zeros((len(m1),len(m1)))
    for ii in range(len(m1)):
        for jj in range(len(m1)):
            if ii == jj:
                mt[ii][jj] = m1[ii][jj]
            if ii != jj:
                mt[ii][jj] = m1[jj][ii]
    return np.array(mt)


def mxadd(m2, m1):
    """matrix addition
    :param m2: left matrix
    :param m1: right matrix
    :return m_out: matrix m_out = m2 + m1
    """
    m_out = np.zeros((len(m2),len(m1)))
    for ii in range(len(m2)):
        for jj in range(len(m1)):
            m_out[ii][jj] = m2[ii][jj]+m1[ii][jj]
    return np.array(m_out)


def mxsub(m2, m1):
    """matrix subtraction
    :param m2: left matrix
    :param m1: right matrix
    :return m_out: matrix m_out = m2 - m1
    """
    m_out = np.zeros((len(m2),len(m1)))
    for ii in range(len(m2)):
        for jj in range(len(m1)):
            m_out[ii][jj] = m2[ii][jj]-m1[ii][jj]
    return np.array(m_out)


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
    return np.array(m_out)


def mxv(m1, v1):
    """multiplies vector by a matrix; currently for 3x3, 4x4 matrices
    :param m1: nxn matrix to be multiplied
    :param v1: vector of n rows to be multiplied
    :return vout: resultant vector v_out = [m1]*v1
    """
    v_out = np.zeros(len(v1))
    if len(v1) == 4:
        for ii in range(len(v1)):
            v_out[ii] = m1[ii][0]*v1[0] + m1[ii][1]*v1[1] + m1[ii][2]*v1[2] + m1[ii][3]*v1[3]
        return np.array(v_out)
    if len(v1) == 3:
        for ii in range(len(v1)):
            v_out[ii] = m1[ii][0]*v1[0] + m1[ii][1]*v1[1] + m1[ii][2]*v1[2]
        return np.array(v_out)


def mxs(scalar, m1):
    """scales a matrix by a scalar
    :param scalar: scalar component to multiply
    :param m1: nxn matrix being scaled
    :return m_out: resultant matrix m_out = scalar*[m1]
    """
    m_out = np.zeros((len(m1),len(m1)))
    for ii in range(len(m1)):
        for jj in range(len(m1)):
            m_out[ii][jj] = scalar * m1[ii][jj]
    return np.array(m_out)


def skew(v1):
    """generates a skewed cross-product matrix from a vector
    :param v1: vector to skew
    :return v_tilde: skewed matrix for v1
    """
    v_tilde = np.zeros((len(v1), len(v1)))
    v_tilde[0][1] = -v1[2]
    v_tilde[0][2] = v1[1]
    v_tilde[1][0] = v1[2]
    v_tilde[1][2] = -v1[0]
    v_tilde[2][0] = -v1[1]
    v_tilde[2][1] = v1[0]
    return np.array(v_tilde)

    
if __name__ == '__main__':
    
    pass