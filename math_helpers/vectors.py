#!/usr/bin/env python3
import numpy as np


def vxadd(v1, v2):
    """vector addition
    :param v1: first vector
    :param v2: second vector
    :return v_out: v_out = v1 + v2
    """
    v_out = np.zeros(len(v1))
    for ii in range(len(v1)):
        v_out[ii] = v1[ii]+v2[ii]
    return np.array(v_out)


def vdotv(v1, v2):
    """vector dot product
    :param v1: first vector
    :param v2: second vector
    :return: v_out = sum of (v1[ii] * v2[ii])
    """
    v_out = 0
    for ii in range(len(v1)):
        v_out += v1[ii]*v2[ii]
    return np.array(v_out)


def norm(v1):
    """returns the magnitude of the two vectors
    in work
    """
    return np.array(np.linalg.norm(v1))


def vcrossv(v1, v2):
    """cross product of two vectors of length 3
    :param v1: first vector (top)
    :param v2: second vector (bottom)
    :return v_out: v_out = v1xv2
    """
    v_out = np.zeros(len(v1))
    v_out[0] = v1[1]*v2[2] - v1[2]*v2[1]
    v_out[1] = -(v1[0]*v2[2] - v1[2]*v2[0])
    v_out[2] = v1[0]*v2[1] - v1[1]*v2[0]
    return np.array(v_out)


def vxvT(v1, v2):
    """vector multiplication
    :param v1: first row vector;
    :param v2: second row vector;
    :return m_out: matrix representing vector product;
                   m_out = v1*v2.T
    """
    m_out = np.zeros((len(v1), len(v1)))
    for ii in range(len(v1)):
        for jj in range(len(v1)):
            m_out[ii][jj] = v1[ii]*v2[jj]
    return np.array(m_out)


def v_cross(v1):
    """returns skew symmetric matrix
    """
    v_tilde = [[    0., -v1[2],  v1[1]],
               [ v1[2],     0., -v1[0]],
               [-v1[1],  v1[0],     0]]
    return np.array(v_tilde)


def vxs(scalar, v1):
    """scales a vector by a scalar
    :param scalar: scalar component to multiply
    :param v1: vector to be scaled
    :return v_out: resultant vector; v_out = scalar*v1
    """
    v_out = np.zeros((len(v1)))
    for ii in range(len(v1)):
        v_out[ii] = scalar * v1[ii]
    return np.array(v_out)


def get_dir_cosines(v1):
    """returns the directional cosines for a given vector
    :param v1: unit vector:
    :return: cosines and sines of vector;
    in work
    """
    cosa = v1[0]
    cosb = v1[1]
    cosc = v1[2]
    sina = np.sqrt(v1[1]**2+v1[2]**2)
    sinb = np.sqrt(v1[2]**2+v1[0]**2)
    sinc = np.sqrt(v1[0]**2+v1[1]**2)
    return cosa, cosb, cosc, sina, sinb, sinc

if __name__ == "__main__":
    
    pass