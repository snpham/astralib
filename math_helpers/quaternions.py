#!/usr/bin/env python3
import os, sys
import numpy as np
from math_helpers import matrices
from math_helpers import rotations
from numpy.linalg import norm


def prv2dcm(e1, e2, e3, theta):
    """converts a principle rotation vector with an angle to a dcm
    """
    dcm = np.zeros([3,3])
    z = 1.0 - np.cos(theta)
    dcm = [[e1*e1*z+np.cos(theta), e1*e2*z+e3*np.sin(theta), e1*e3*z-e2*np.sin(theta)],
           [e2*e1*z-e3*np.sin(theta), e2*e2*z+np.cos(theta), e2*e3*z+e1*np.sin(theta)],
           [e3*e1*z+e2*np.sin(theta), e3*e2*z-e1*np.sin(theta), e3*e3*z+np.cos(theta)]]
    return dcm


def prv_angle(dcm):
    """compute the principle rotation angle from a dcm
    """
    theta = np.arccos(1./2. * (dcm[0][0] + dcm[1][1] + dcm[2][2] - 1))
    return theta


def prv_axis(dcm):
    """compute the principle rotation vector from a dcm
    """
    theta = prv_angle(dcm)
    factor = 1./(2.*np.sin(theta))
    e1 = factor * (dcm[1][2] - dcm[2][1])
    e2 = factor *(dcm[2][0] - dcm[0][2])
    e3 = factor *(dcm[0][1] - dcm[1][0])
    return e1, e2, e3

    
def quat2dcm(qset):
    """converts a quaternion set into a dcm
    """
    s1, q1, q2, q3 = qset
    dcm = np.zeros([3,3])
    dcm = [[s1*s1+q1*q1-q2*q2-q3*q3, 2*(q1*q2+s1*q3), 2*(q1*q3-s1*q2)],
           [2*(q1*q2-s1*q3), s1*s1-q1*q1+q2*q2-q3*q3, 2*(q2*q3+s1*q1)],
           [2*(q1*q3+s1*q2), 2*(q2*q3-s1*q1), s1*s1-q1*q1-q2*q2+q3*q3]]
    return dcm


def dcm2quat(dcm):
    """converts a dcm into a quaternion set; note singularties exist
    when s1=0, use sheppard's method if possible
    """
    trace = dcm[0][0] + dcm[1][1] + dcm[2][2]
    s1 = 1./2. * np.sqrt(trace + 1)
    b1 = (dcm[1][2]-dcm[2][1]) / (4.*s1)
    b2 = (dcm[2][0]-dcm[0][2]) / (4.*s1)
    b3 = (dcm[0][1]-dcm[1][0]) / (4.*s1)


def dcm2quat_sheppard(dcm):
    """converts a dcm to quaternions based on sheppard's method;
    see pg 110, Analytical Mechanics of Space Systems
    """
    trace = dcm[0][0] + dcm[1][1] + dcm[2][2]
    s1_sq = 1./4. * (1+trace)
    b1_sq = 1./4. * (1+2*dcm[0][0]-trace)
    b2_sq = 1./4. * (1+2*dcm[1][1]-trace)
    b3_sq = 1./4. * (1+2*dcm[2][2]-trace)
    quats = [s1_sq, b1_sq, b2_sq, b3_sq]
    print(quats)
    if np.argmax(quats) == 0:
        s1 = np.sqrt(s1_sq)
        b1 = (dcm[1][2]-dcm[2][1])/(4.*s1)
        b2 = (dcm[2][0]-dcm[0][2])/(4.*s1)
        b3 = (dcm[0][1]-dcm[1][0])/(4.*s1)
    elif np.argmax(quats) == 1:
        b1 = np.sqrt(b1_sq)
        s1 = (dcm[1][2]-dcm[2][1])/(4.*b1)
        b2 = (dcm[0][1]+dcm[1][0])/(4.*b1)
        b3 = (dcm[2][0]+dcm[0][2])/(4.*b1)
    elif np.argmax(quats) == 2:
        b2 = np.sqrt(b2_sq)
        s1 = (dcm[2][0]-dcm[0][2])/(4.*b2)
        b1 = (dcm[0][1]+dcm[1][0])/(4.*b2)
        b3 = (dcm[1][2]+dcm[2][1])/(4.*b2)
    elif np.argmax(quats) == 3:
        b3 = np.sqrt(b3_sq)
        s1 = (dcm[0][1]-dcm[1][0])/(4.*b3)
        b1 = (dcm[2][0]+dcm[0][2])/(4.*b3)
        b2 = (dcm[1][2]+dcm[2][1])/(4.*b3)
    return [s1, b1, b2, b3]


def euler2quat(a1, a2, a3, sequence='313'):
    """returns quaternion set from an euler sequence, currently only
    accepts 313 sequence
    """
    if sequence == '313':
        s1 = np.cos(a2/2.) * np.cos((a3+a1)/2.)
        b1 = np.sin(a2/2.) * np.cos((a3-a1)/2.)
        b2 = np.sin(a2/2.) * np.sin((a3-a1)/2.)
        b3 = np.cos(a2/2.) * np.sin((a3+a1)/2.)
    return [s1, b1, b2, b3]


def qxq(q1st, q2nd):
    """computes the quaternion for a composite rotation of two
    quaternion sets, v_out = [mq1st][vq2nd]
    """
    s1, b1, b2, b3 = q2nd
    matrix = [[ s1, -b1, -b2, -b3],
              [ b1,  s1,  b3, -b2],
              [ b2, -b3,  s1,  b1],
              [ b3,  b2, -b1,  s1]]
    v_out = matrices.mxv(matrix, q1st)
    return v_out


def qxq_transmute(q1st, q2nd):
    s1, b1, b2, b3 = q1st
    matrix = [[ s1, -b1, -b2, -b3],
              [ b1,  s1, -b3,  b2],
              [ b2,  b3,  s1, -b1],
              [ b3, -b2,  b1,  s1]]
    v_out =  matrices.mxv(matrix, q1st)    
    return v_out


def quat_kde_fromq(qset, wset):
    w1, w2, w3 = 0.5 * wset
    matrix = [[0, -w1, -w2, -w3],
              [w1,  0,  w3, -w2],
              [w2,-w3,   0,  w1],
              [w3, w2, -w1,   0]]
    return  matrices.mxv(matrix, qset)


def quat_kde_fromw(qset, wset):
    s1, b1, b2, b3 = 0.5 * qset
    matrix = [[s1,-b1,-b2,-b3],
              [b1, s1,-b3, b2],
              [b2, b3, s1,-b1],
              [b3,-b2, b1, s1]]
    wset = [0, wset]
    return  matrices.mxv(matrix, wset)


def quat2mrp(qset):
    """computes the modified rodrigues parameters from quaternion sets
    """
    sigma1 = qset[1] / (1.+qset[0])
    sigma2 = qset[2] / (1.+qset[0])
    sigma3 = qset[3] / (1.+qset[0])
    return [sigma1, sigma2, sigma3]


def quat2mrps(qset):
    """computes the shadow set modified rodrigues parameters from
    quaternion sets
    """
    sigma1 = -qset[1] / (1.-qset[0])
    sigma2 = -qset[2] / (1.-qset[0])
    sigma3 = -qset[3] / (1.-qset[0])
    return [sigma1, sigma2, sigma3]


def mrp2dcm(sigmaset):
    """Transforms a set of modified rodrigues parameters into a dcm
    """
    imatrix = np.eye(3)
    sigma_skewmat =  matrices.skew_tilde(v1=sigmaset)
    # sigma_skewmat_sq = np.dot(sigma_skewmat,sigma_skewmat)
    sigma_skewmat_sq =  matrices.mxm(m2=sigma_skewmat, m1=sigma_skewmat)
    # amat = np.dot(8.0, sigma_skewmat_sq)
    amat =  matrices.mxscalar(scalar=8.0, m1=sigma_skewmat_sq)
    bscalar = 4.0 * ( 1 - norm(sigmaset)**2)
    # bmat = np.dot(bscalar, sigma_skewmat)
    bmat =  matrices.mxscalar(scalar=bscalar, m1=sigma_skewmat)
    cscalar = 1.0 / ((1.0 + norm(sigmaset)**2)**2)
    asubb =  matrices.mxsub(m2=amat, m1=bmat)
    # dcm = np.dot(cscalar, asubb)
    dcm =  matrices.mxscalar(scalar=cscalar, m1=asubb)
    return dcm


if __name__ == "__main__":
    from pprint import pprint as pp
    # testing principle rotation parameters
    # deg = np.deg2rad([10, 25, -15])
    # matrix = rotations.rotate_euler(deg[0], deg[1], deg[2], '321')
    # angle = np.rad2deg(prv_angle(matrix))
    # axis = prv_axis(matrix)
    # print(matrix)
    # print(angle)
    # print(axis)
    # print(angle - 360.)

    # testing quaternions
    # b1, b2, b3 = np.deg2rad([30, -45, 60])

    # brotate = rotations.rotate_euler(b1, b2, b3, '321')
    #     #print(f'BN: {brotate}')

    # f1, f2, f3 = np.deg2rad([10., 25., -15.])
    # frotate = rotations.rotate_euler(f1, f2, f3, '321')
    # frot = rotations.rotate_sequence(f1, f2, f3, '321')
    # print(f'FN: {frotate}')
    # quats = dcm2quat_sheppard(frotate)
    # print(quats)

    # a1, a2, a3 = rotations.dcm_inverse(frotate, sequence='313')
    # print(np.rad2deg(a1), np.rad2deg(a2), np.rad2deg(a3))
    # quat = euler2quat(a1, a2 , a3, sequence='313')
    # print(quat)

    # testing qxq and transmutation
    # q1st = [0., 1./np.sqrt(2.), 1./np.sqrt(2.), 0.]
    # q2nd = [0.5*np.sqrt(np.sqrt(3)/2+1), -0.5*np.sqrt(np.sqrt(3)/2+1), -np.sqrt(2)/(4*np.sqrt(2+np.sqrt(3))), np.sqrt(2)/(4*np.sqrt(2+np.sqrt(3)))]
    # # print(q1st)
    # # print(q2nd)
    # v_out = qxq(q1st, q2nd)
    # v_actual = [1/(2*np.sqrt(2))*np.sqrt(3), 1/(2*np.sqrt(2))*np.sqrt(3), 1/(2*np.sqrt(2))*1, 1/(2*np.sqrt(2))*1]
    # print(v_out)
    # print(v_actual)
    # dcm = quat2dcm(v_out)
    # dcm_actual = quat2dcm(v_actual)
    # pp(dcm)
    # pp(dcm_actual)

    # #testing mrp's
    # qset = [0.961798, -0.14565, 0.202665, 0.112505]
    # sigmaset = quat2mrp(qset)
    # sigmasets = quat2mrps(qset)
    # print(sigmaset, sigmasets)