#!/usr/bin/env python3
from math_helpers import matrices, vectors
from numpy.linalg import norm
import numpy as np


def rotate_x(angle):
    """rotation about the x axis
    :param angle: magnitude of angle of rotation (radians)
    :return matrix: rotation matrix of "angle" about x-axis 
    """
    matrix = [[1.0,            0.0,           0.0],
              [0.0,  np.cos(angle), np.sin(angle)],
              [0.0, -np.sin(angle), np.cos(angle)]]
    return matrix


def rotate_y(angle):
    """rotation about the y axis
    :param angle: magnitude of angle of rotation (radians)
    :return matrix: rotation matrix of "angle" about y-axis 
    """
    matrix = [[np.cos(angle), 0.0, -np.sin(angle)],
              [          0.0, 1.0,            0.0],
              [np.sin(angle), 0.0,  np.cos(angle)]]
    return matrix


def rotate_z(angle):
    """rotation about the z axis
    :param angle: magnitude of angle of rotation (radians)
    :return matrix: rotation matrix of "angle" about z-axis 
    """
    matrix = [[ np.cos(angle), np.sin(angle), 0.0],
              [-np.sin(angle), np.cos(angle), 0.0],
              [           0.0,           0.0, 1.0]]
    return matrix


def dcm_rate(omega_tilde, dcm):
    return -matrices.mxm(omega_tilde, dcm)


def dcm_inverse(dcm, sequence='321'):
    if sequence == '321':
        angle1st = np.arctan2(dcm[0][1], dcm[0][0])
        angle2nd = -np.arcsin(dcm[0][2])
        angle3rd = np.arctan2(dcm[1][2], dcm[2][2])
    if sequence == '313':
        angle1st = np.arctan2(dcm[2][0], -dcm[2][1])
        angle2nd = np.arccos(dcm[2][2])
        angle3rd = np.arctan2(dcm[0][2], dcm[1][2])
    return angle1st, angle2nd, angle3rd


def rotate_sequence(a1st, a2nd, a3rd, sequence='321'):
    if sequence == '321':
        matrix = [[np.cos(a2nd)*np.cos(a1st), np.cos(a2nd)*np.sin(a1st), -np.sin(a2nd)],
                  [np.sin(a3rd)*np.sin(a2nd)*np.cos(a1st)-np.cos(a3rd)*np.sin(a1st), np.sin(a3rd)*np.sin(a2nd)*np.sin(a1st)+np.cos(a3rd)*np.cos(a1st), np.sin(a3rd)*np.cos(a2nd)],
                  [np.cos(a3rd)*np.sin(a2nd)*np.cos(a1st)+np.sin(a3rd)*np.sin(a1st), np.cos(a3rd)*np.sin(a2nd)*np.sin(a1st)-np.sin(a3rd)*np.cos(a1st), np.cos(a3rd)*np.cos(a2nd)]]
    if sequence == '313':
        matrix = [[ np.cos(a3rd)*np.cos(a1st)-np.sin(a3rd)*np.cos(a2nd)*np.sin(a1st),  np.cos(a3rd)*np.sin(a1st)+np.sin(a3rd)*np.cos(a2nd)*np.cos(a1st), np.sin(a3rd)*np.sin(a2nd)],
                  [-np.sin(a3rd)*np.cos(a1st)-np.cos(a3rd)*np.cos(a2nd)*np.sin(a1st), -np.sin(a3rd)*np.sin(a1st)+np.cos(a3rd)*np.cos(a2nd)*np.cos(a1st), np.cos(a3rd)*np.sin(a2nd)],
                  [np.sin(a2nd)*np.sin(a1st), -np.sin(a2nd)*np.cos(a1st), np.cos(a2nd)]]
    return matrix


def rotate_euler(a1, a2, a3, sequence='321'):
    if sequence == '321':
        product1 = matrices.mxm(m2=rotate_y(a2), m1=rotate_z(a1))
        product2 = matrices.mxm(m2=rotate_x(a3), m1=product1)
    if sequence == '313':
        product1 = matrices.mxm(m2=rotate_y(a2), m1=rotate_z(a1))
        product2 = matrices.mxm(m2=rotate_z(a3), m1=product1)
    return product2


def axisofr(Tmatrix):
    vout = np.zeros(3)
    vout[0] = Tmatrix[0][1]*Tmatrix[1][2] - (Tmatrix[1][1] -1)*Tmatrix[0][2]
    vout[1] = Tmatrix[1][0]*Tmatrix[0][2] - (Tmatrix[0][0] -1)*Tmatrix[1][2]
    vout[2] = (Tmatrix[0][0] - 1)*(Tmatrix[1][1] - 1) - Tmatrix[0][1]*Tmatrix[1][0]
    phi = np.arccos((Tmatrix[0][0]+Tmatrix[1][1]+Tmatrix[2][2] - 1) / 2.)
    return vout, phi


def triad(v1, v2):
    """TRIAD attitude estimation method. Uses two vector observations, 
    v1 and v2, to establish a frame tset with v1 being primary and
    v2 secondary.
    """
    t1 = v1
    v1xv2 = vectors.vcrossv(v1, v2)
    t2 = v1xv2 / norm(v1xv2)
    t3 = vectors.vcrossv(t1, t2)
    tset = [t1, t2, t3]
    # transform matrix for new tset frame to v frame
    t2vmatrix = matrices.mtranspose(tset)
    return tset, t2vmatrix


def davenportq(vset, n):
    pass



if __name__ == "__main__":
    #testing triad method
    v1 = [1, 0, 0]
    v2 = [0, 0, 1]
    tveci, t2i = triad(v1, v2)
    # print(tveci)
    # print(t2i)
    bn_actual = matrices.rotate_euler(a1=np.deg2rad(30), a2=np.deg2rad(20), a3=np.deg2rad(-10), 
                               sequence='321')
    v1out_a = matrices.mxv(bn_actual, v1)
    v2out_a = matrices.mxv(bn_actual, v2)
    v1out = [0.8190, -0.5282, 0.2242]
    v2out = [-0.3138, -0.1584, 0.9362]
    # print(v1out, v2out)
    tvec, tmatrix = triad(v1out, v2out)
    # print(tvec)
    # print(tmatrix)
    bn = matrices.matrix_multT(m2=tmatrix, m1=t2i)
    print(bn)
    bn_error = matrices.matrix_multT(m2=bn, m1=bn_actual)
    print(bn_error)