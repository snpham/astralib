#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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


def T_ijk2topo(lon, lat, frame='sez'):
    if frame == 'sez':
        m1 = rotate_z(lon)
        m2 = rotate_y(lat)
        mat = matrices.mxm(m2=m2, m1=m1)
    return mat


def T_pqw2ijk(raan, incl, argp):
    s, c = np.sin, np.cos
    rot_mat = [[c(raan)*c(argp)-s(raan)*s(argp)*c(incl), -c(raan)*s(argp)-s(raan)*c(argp)*c(incl), s(raan)*s(incl)],
               [s(raan)*c(argp)+c(raan)*s(argp)*c(incl), -s(raan)*s(argp)+c(raan)*c(argp)*c(incl), -c(raan)*s(incl)],
               [s(argp)*s(incl), c(argp)*s(incl), c(incl)]]
    return rot_mat


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


def rotate_sequence(a1, a2, a3, sequence='321'):
    s, c = np.sin, np.cos

    if sequence == '321':
        matrix = [[c(a2)*c(a1),                   c(a2)*s(a1),                  -s(a2)],
                  [s(a3)*s(a2)*c(a1)-c(a3)*s(a1), s(a3)*s(a2)*s(a1)+c(a3)*c(a1), s(a3)*c(a2)],
                  [c(a3)*s(a2)*c(a1)+s(a3)*s(a1), c(a3)*s(a2)*s(a1)-s(a3)*c(a1), c(a3)*c(a2)]]
    if sequence == '313':
        matrix = [[ c(a3)*c(a1)-s(a3)*c(a2)*s(a1),  c(a3)*s(a1)+s(a3)*c(a2)*c(a1), s(a3)*s(a2)],
                  [-s(a3)*c(a1)-c(a3)*c(a2)*s(a1), -s(a3)*s(a1)+c(a3)*c(a2)*c(a1), c(a3)*s(a2)],
                  [ s(a2)*s(a1),                   -s(a2)*c(a1),                   c(a2)]]
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


def wvec_frm_eulerrates_o2b(aset, rates, sequence='321'):
    """orbit to body frame; in work
    """
    s, c = np.sin, np.cos

    if sequence == '321':
        matrix = [[-s(aset[1]),            0.0,        1.0],
                  [s(aset[2])*c(aset[1]),  c(aset[2]), 0.0],
                  [c(aset[2])*c(aset[1]), -s(aset[2]), 0.0]]
    return matrices.mxv(m1=matrix, v1=rates)


def eulerrates_frm_wvec_o2b(aset, wvec, sequence='321'):
    """orbit to body frame; in work
    """
    s, c = np.sin, np.cos
    
    if sequence == '321':
        matrix = [[0.0,        s(aset[2]),             c(aset[2])],
                  [0.0,        c(aset[2])*c(aset[1]), -s(aset[2])*c(aset[1])],
                  [c(aset[1]), s(aset[2])*s(aset[1]),  c(aset[2])*s(aset[1])]]
        mxwvec = matrices.mxv(m1=matrix, v1=wvec)
    return vectors.vxscalar(scalar=1/c(aset[1]), v1=mxwvec)


def wvec_frm_eulerrates_n2b(aset, rates, Omega, sequence='321'):
    """inertial to orbit to body frame; o2 = orbit normal; in work
    """
    if sequence == '321':
        w_o2b = wvec_frm_eulerrates_o2b(aset=aset, rates=rates, sequence='321')
        eulerdcm = rotate_sequence(aset[0], aset[1], aset[2], sequence='321')
        # print(matrices.mtranspose(eulerdcm)[1])
        w_n2o = vectors.vxscalar(scalar=Omega, v1=matrices.mtranspose(eulerdcm)[1])

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
    #testing triad method
    # v1 = [1, 0, 0]
    # v2 = [0, 0, 1]
    # tveci, t2i = triad(v1, v2)
    # # print(tveci)
    # # print(t2i)
    # bn_actual = rotate_euler(a1=np.deg2rad(30), a2=np.deg2rad(20), a3=np.deg2rad(-10), 
    #                            sequence='321')
    # v1out_a = matrices.mxv(bn_actual, v1)
    # v2out_a = matrices.mxv(bn_actual, v2)
    # v1out = [0.8190, -0.5282, 0.2242]
    # v2out = [-0.3138, -0.1584, 0.9362]
    # # print(v1out, v2out)
    # tvec, tmatrix = triad(v1out, v2out)
    # # print(tvec)
    # # print(tmatrix)
    # bn = matrices.mxm(m2=tmatrix, m1=t2i)
    # print(bn)
    # bn_error = matrices.mxm(m2=bn, m1=matrices.mtranspose(bn_actual))
    # print(bn_error)

    aset = [1.0, 0.0, 0.0]
    rates = [1.0, 0.0, 0.0]
    sequence = '321'
    w_n2b = wvec_frm_eulerrates_n2b(aset=aset, rates=rates, Omega=0.0, sequence='321')
    print(w_n2b)
    wvec = [1.0, 0.0, 0.0]
    rates_n2b = eulerrates_frm_wvec_n2b(aset=aset, wvec=wvec, Omega=0.0, sequence='321')
    print(rates_n2b)