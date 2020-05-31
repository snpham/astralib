#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from math_helpers import matrices as mat
from math_helpers import vectors, quaternions
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


def euler2dcm(a1st, a2nd, a3rd, sequence='321'):
    """euler rotation sequence utilizing rotate functions
    :param a1st: angle for first rotation axis (rad)
    :param a2nd: angle for second rotation axis (rad)
    :param a3rd: angle for third rotation axis (rad)
    :param sequence: euler sequence; default='321'
    :return: transformation matrix of rotation sequence
    """
    a1, a2, a3 = a1st, a2nd, a3rd
    if sequence == '321':
        T1 = mat.mxm(m2=rotate_y(a2), m1=rotate_z(a1))
        T2 = mat.mxm(m2=rotate_x(a3), m1=T1)
    if sequence == '313':
        T1 = mat.mxm(m2=rotate_y(a2), m1=rotate_z(a1))
        T2 = mat.mxm(m2=rotate_z(a3), m1=T1)
    return T2
    

def euler2dcm2(a1st, a2nd, a3rd, sequence='321'):
    """euler rotation sequence utilizing predefined matrix
    :param a1st: angle for first rotation axis (rad)
    :param a2nd: angle for second rotation axis (rad)
    :param a3rd: angle for third rotation axis (rad)
    :param sequence: euler sequence; default='321'
    :return: transformation matrix of rotation sequence
    """
    s, c = np.sin, np.cos
    a1, a2, a3 = a1st, a2nd, a3rd
    if sequence == '321':
        matrix = [[c(a2)*c(a1),                   c(a2)*s(a1),                  -s(a2)],
                  [s(a3)*s(a2)*c(a1)-c(a3)*s(a1), s(a3)*s(a2)*s(a1)+c(a3)*c(a1), s(a3)*c(a2)],
                  [c(a3)*s(a2)*c(a1)+s(a3)*s(a1), c(a3)*s(a2)*s(a1)-s(a3)*c(a1), c(a3)*c(a2)]]
    if sequence == '313':
        matrix = [[ c(a3)*c(a1)-s(a3)*c(a2)*s(a1),  c(a3)*s(a1)+s(a3)*c(a2)*c(a1), s(a3)*s(a2)],
                  [-s(a3)*c(a1)-c(a3)*c(a2)*s(a1), -s(a3)*s(a1)+c(a3)*c(a2)*c(a1), c(a3)*s(a2)],
                  [ s(a2)*s(a1),                   -s(a2)*c(a1),                   c(a2)]]
    return matrix


def dcm2euler(dcm, sequence='321'):
    """compute euler angles from a rotation matrix
    :param dcm: direction cosine matrix
    :param sequence: euler sequence; default='321'
    :return: angle set for the dcm and euler sequence
    """
    if sequence =='321':
        angle1st = np.arctan2(dcm[0][1],dcm[0][0])
        angle2nd = -np.arcsin(dcm[0][2])
        angle3rd = np.arctan2(dcm[1][2],dcm[2][2])
    if sequence == '313':
        angle1st = np.arctan2(dcm[2][0],-dcm[2][1])
        angle2nd = -np.arccos(dcm[2][2])
        angle3rd = np.arctan2(dcm[0][2],dcm[1][2])
    return angle1st, angle2nd, angle3rd


def prv_angle(dcm):
    """compute the principle rotation angle from a dcm
    :param dcm: direction cosine matrix to extract rotation vector 
                from
    :return phi: principal rotation angle (rad)
    """
    phi = np.arccos(1./2.*(dcm[0][0]+dcm[1][1]+dcm[2][2]-1))
    return phi


def prv_axis(dcm):
    """compute the principle rotation vector from a dcm
    :param dcm: direction cosine matrix to extract rotation vector 
                from
    :return evec: principle rotation vector (eigvec corres. to the
                  eigval of +1)
    :return phi: magnitude of the angle of rotation (radians)
    """
    phi = prv_angle(dcm)
    factor = 1./(2.*np.sin(phi))
    e1 = factor * (dcm[1][2] - dcm[2][1])
    e2 = factor *(dcm[2][0] - dcm[0][2])
    e3 = factor *(dcm[0][1] - dcm[1][0])
    evec = [e1, e2, e3]
    return evec, phi


def axisofr(Tmatrix):
    """in work
    """
    vout = np.zeros(3)
    vout[0] = Tmatrix[0][1]*Tmatrix[1][2] - (Tmatrix[1][1] -1)*Tmatrix[0][2]
    vout[1] = Tmatrix[1][0]*Tmatrix[0][2] - (Tmatrix[0][0] -1)*Tmatrix[1][2]
    vout[2] = (Tmatrix[0][0] - 1)*(Tmatrix[1][1] - 1) - Tmatrix[0][1]*Tmatrix[1][0]
    phi = np.arccos((Tmatrix[0][0]+Tmatrix[1][1]+Tmatrix[2][2] - 1) / 2.)
    return vout, phi


def dcm_inverse(dcm, sequence='321'):
    """in work
    """
    if sequence == '321':
        angle1st = np.arctan2(dcm[0][1], dcm[0][0])
        angle2nd = -np.arcsin(dcm[0][2])
        angle3rd = np.arctan2(dcm[1][2], dcm[2][2])
    if sequence == '313':
        angle1st = np.arctan2(dcm[2][0], -dcm[2][1])
        angle2nd = np.arccos(dcm[2][2])
        angle3rd = np.arctan2(dcm[0][2], dcm[1][2])
    return angle1st, angle2nd, angle3rd


def dcm_rate(omega_tilde, dcm):
    """in work
    """
    return -mat.mxm(omega_tilde, dcm)


def triad(v1, v2):
    """TRIAD attitude estimation method. Uses two vector observations, 
    v1 and v2, to establish a frame tset with v1 being primary and
    v2 secondary.
    :param v1: first inertial vector measurement
    :param v2: second inertial vector meansurement
    return tset: 3x3 matrix with t1, t2, t3 as columns 0, 1, 2;
    return t2vmatrix: transpose of triad frame vectors tset; rows 
                      represents triad vectors in terms of inertial
                      basis vectors
    """
    t1 = np.array(v1)
    v1xv2 = vectors.vcrossv(v1, v2)
    t2 = v1xv2 / norm(v1xv2)
    t3 = vectors.vcrossv(t1, t2)
    tset = [t1, t2, t3]
    # transform matrix for new tset frame to v frame
    t2vmatrix = mat.mtranspose(tset)
    return tset, t2vmatrix


def davenportq(vset_nrtl, vset_body, weights, sensors=2):
    """solving Wahba's problem using quaternions; see pg 147
    :param vset_nrtl: inertial sensor measurement vectors
    :param vset_body: sensor measurement vectors in body frame
    :param weights: weights for each sensor measurement
    :sensors: number of sensors computed; default=2
    :return qset: quaternion set for optimal attitude estimate
                  satisfying Wahba's problem
    """
    B = np.zeros((3,3))
    for s in range(sensors):
        m1 = mat.mxscalar(scalar=weights[s], \
                          m1=vectors.vxv(v1=vset_body[s], v2=vset_nrtl[s]))
        B = mat.mxadd(m2=m1, m1=B)
    S = mat.mxadd(m2=B, m1=mat.mtranspose(m1=B))
    sigma = B[0][0] + B[1][1] + B[2][2]
    Z = [[B[1][2]-B[2][1]], [B[2][0]-B[0][2]], [B[0][1]-B[1][0]]]
    K = np.zeros((4,4))
    K[0][0] = sigma
    Kterm4 = mat.mxsub(m2=S, m1=mat.mxscalar(scalar=sigma, m1=np.eye(3)))
    for ii in range(3):
        K[0][ii+1] = Z[ii][0]
        K[ii+1][0] = Z[ii][0]
        for jj in range(3):
            K[ii+1][jj+1] = Kterm4[ii][jj]
    eigvals, eigvecs = np.linalg.eig(K)
    qset = eigvecs[:, np.argmax(eigvals)]
    dcm = quaternions.quat2dcm(qset=qset)
    return qset


def wvec_frm_eulerrates_o2b(aset, rates, sequence='321'):
    """orbit to body frame; in work
    """
    s, c = np.sin, np.cos

    if sequence == '321':
        matrix = [[-s(aset[1]),            0.0,        1.0],
                  [s(aset[2])*c(aset[1]),  c(aset[2]), 0.0],
                  [c(aset[2])*c(aset[1]), -s(aset[2]), 0.0]]
    return mat.mxv(m1=matrix, v1=rates)


def eulerrates_frm_wvec_o2b(aset, wvec, sequence='321'):
    """orbit to body frame; in work
    """
    s, c = np.sin, np.cos
    
    if sequence == '321':
        matrix = [[0.0,        s(aset[2]),             c(aset[2])],
                  [0.0,        c(aset[2])*c(aset[1]), -s(aset[2])*c(aset[1])],
                  [c(aset[1]), s(aset[2])*s(aset[1]),  c(aset[2])*s(aset[1])]]
        mxwvec = mat.mxv(m1=matrix, v1=wvec)
    return vectors.vxscalar(scalar=1/c(aset[1]), v1=mxwvec)


def wvec_frm_eulerrates_n2b(aset, rates, Omega, sequence='321'):
    """inertial to orbit to body frame; o2 = orbit normal; in work
    """
    if sequence == '321':
        w_o2b = wvec_frm_eulerrates_o2b(aset=aset, rates=rates, sequence='321')
        eulerdcm = euler2dcm(aset[0], aset[1], aset[2], sequence='321')
        w_n2o = vectors.vxscalar(scalar=Omega, v1=mat.mtranspose(eulerdcm)[1])

    return vectors.vxadd(v1=w_o2b, v2=w_n2o)


def eulerrates_frm_wvec_n2b(aset, wvec, Omega, sequence='321'):
    """inertial to orbit to body frame; o2 = orbit normal; in work
    """
    if sequence == '321':
        rates_o2b = eulerrates_frm_wvec_o2b(aset=aset, wvec=wvec, sequence='321')
        term2 = vectors.vxscalar(scalar=Omega/np.cos(aset[1]), \
            v1=[np.sin(aset[1])*np.sin(aset[0]), np.cos(aset[1])*np.cos(aset[0]), np.sin(aset[0])])

    return vectors.vxadd(v1=rates_o2b, v2=-term2)


if __name__ == "__main__":
    pass
    # #testing triad method
    # v1 = [1, 0, 0]
    # v2 = [0, 0, 1]
    # tveci, t2i = triad(v1, v2)
    # print(tveci)
    # print(t2i)
    # # bn_actual = euler2dcm(a1=np.deg2rad(30), a2=np.deg2rad(20), a3=np.deg2rad(-10), 
    # #                            sequence='321')
    # # v1out_a = mat.mxv(bn_actual, v1)
    # # v2out_a = mat.mxv(bn_actual, v2)
    # v1out = [0.8190, -0.5282, 0.2242]
    # v2out = [-0.3138, -0.1584, 0.9362]
    # # print(v1out, v2out)
    # tvec, tmatrix = triad(v1out, v2out)
    # # print(tvec)
    # # print(tmatrix)
    # bn = mat.mxm(m2=tmatrix, m1=t2i)
    # print(bn)
    # bn_error = mat.mxm(m2=bn, m1=mat.mtranspose(bn_actual))
    # print(bn_error)

    # aset = [1.0, 0.0, 0.0]
    # rates = [1.0, 0.0, 0.0]
    # sequence = '321'
    # w_n2b = wvec_frm_eulerrates_n2b(aset=aset, rates=rates, Omega=0.0, sequence='321')
    # print(w_n2b)
    # wvec = [1.0, 0.0, 0.0]
    # rates_n2b = eulerrates_frm_wvec_n2b(aset=aset, wvec=wvec, Omega=0.0, sequence='321')
    # print(rates_n2b)
