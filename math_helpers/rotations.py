#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from math_helpers import matrices as mat
from math_helpers import vectors as vec
from math_helpers import quaternions as quat
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
    return np.array(matrix)


def rotate_y(angle):
    """rotation about the y axis
    :param angle: magnitude of angle of rotation (radians)
    :return matrix: rotation matrix of "angle" about y-axis 
    """
    matrix = [[np.cos(angle), 0.0, -np.sin(angle)],
              [          0.0, 1.0,            0.0],
              [np.sin(angle), 0.0,  np.cos(angle)]]
    return np.array(matrix)


def rotate_z(angle):
    """rotation about the z axis
    :param angle: magnitude of angle of rotation (radians)
    :return matrix: rotation matrix of "angle" about z-axis 
    """
    matrix = [[ np.cos(angle), np.sin(angle), 0.0],
              [-np.sin(angle), np.cos(angle), 0.0],
              [           0.0,           0.0, 1.0]]
    return np.array(matrix)


def euler2dcm(angles, sequence='321'):
    """euler rotation sequence utilizing rotate functions
    :param a1st: angle for first rotation axis (rad)
    :param a2nd: angle for second rotation axis (rad)
    :param a3rd: angle for third rotation axis (rad)
    :param sequence: euler sequence; default='321'
    :return: transformation matrix of rotation sequence
    """
    a1, a2, a3 = angles[0], angles[1], angles[2]
    if sequence == '321':
        T1 = mat.mxm(m2=rotate_y(a2), m1=rotate_z(a1))
        T2 = mat.mxm(m2=rotate_x(a3), m1=T1)
    elif sequence == '313':
        T1 = mat.mxm(m2=rotate_y(a2), m1=rotate_z(a1))
        T2 = mat.mxm(m2=rotate_z(a3), m1=T1)
    else:
         raise ValueError(f'euler sequence not yet implemented')
    return np.array(T2)
    

def euler2dcm2(angles, sequence='321'):
    """euler rotation sequence utilizing predefined matrix
    :param a1st: angle for first rotation axis (rad)
    :param a2nd: angle for second rotation axis (rad)
    :param a3rd: angle for third rotation axis (rad)
    :param sequence: euler sequence; default='321'
    :return: transformation matrix of rotation sequence
    """
    s, c = np.sin, np.cos
    a1, a2, a3 =  angles[0], angles[1], angles[2]
    if sequence == '321':
        matrix = [[c(a2)*c(a1),                   c(a2)*s(a1),                  -s(a2)],
                  [s(a3)*s(a2)*c(a1)-c(a3)*s(a1), s(a3)*s(a2)*s(a1)+c(a3)*c(a1), s(a3)*c(a2)],
                  [c(a3)*s(a2)*c(a1)+s(a3)*s(a1), c(a3)*s(a2)*s(a1)-s(a3)*c(a1), c(a3)*c(a2)]]
    elif sequence == '313':
        matrix = [[ c(a3)*c(a1)-s(a3)*c(a2)*s(a1),  c(a3)*s(a1)+s(a3)*c(a2)*c(a1), s(a3)*s(a2)],
                  [-s(a3)*c(a1)-c(a3)*c(a2)*s(a1), -s(a3)*s(a1)+c(a3)*c(a2)*c(a1), c(a3)*s(a2)],
                  [ s(a2)*s(a1),                   -s(a2)*c(a1),                   c(a2)]]
    else:
        raise ValueError(f'euler sequence not yet implemented')           
    return np.array(matrix)


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
    elif sequence == '313':
        angle1st = np.arctan2(dcm[2][0],-dcm[2][1])
        angle2nd = np.arccos(dcm[2][2])
        angle3rd = np.arctan2(dcm[0][2],dcm[1][2])
    else:
        raise ValueError(f'euler sequence not yet implemented')
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
    evec = np.array([e1, e2, e3])
    return evec, phi


def axisofr(Tmatrix):
    """in work
    """
    vout = np.array([Tmatrix[0][1]*Tmatrix[1][2] - (Tmatrix[1][1] -1)*Tmatrix[0][2],
                     Tmatrix[1][0]*Tmatrix[0][2] - (Tmatrix[0][0] -1)*Tmatrix[1][2],
                    (Tmatrix[0][0] - 1)*(Tmatrix[1][1] - 1) - Tmatrix[0][1]*Tmatrix[1][0]])
    phi = prv_angle(Tmatrix)
    return vout, phi


def dcm_rate(omega_tilde, dcm):
    """returns the DCM differential kinematic equation showing how
    the dcm evolve over time.
    :param omega_tilde: body angular velocity vector in skewed
                        symmetric matrix form
    :param dcm: direction cosine matrix of a rotation
    :return dcm_dot = -[~w][dcm]
    """
    return np.array(-mat.mxm(omega_tilde, dcm))


def triad(v1, v2):
    """TRIAD attitude estimation method. Uses two vector observations, 
    v1 and v2, to establish a frame tset with v1 being primary and
    v2 secondary. Resultant frame will be same as input v frame.
    :param v1: primary vector measurement in v frame
    :param v2: secondary vector meansurement in v frame
    return tset: t1, t2, t3 in the v frame;
    return AT_matrix: transpose of triad frame vectors tset; each
                      column represent each t vector in the v frame;
                      i.e.: [BbarT] = [t1bar_B t2bar_B t3bar_B]
                            [NT] = [t1_N t2_N t3_N]
    """
    t1 = np.array(v1)
    t2 = vec.vcrossv(v1, v2) / norm(vec.vcrossv(v1, v2))
    t3 = vec.vcrossv(t1, t2)
    tset = [t1, t2, t3]
    # transform to v frame matrix
    AT_matrix = np.array(mat.mT(tset))
    return tset, AT_matrix


def davenportq(vset_nrtl, vset_body, weights, sensors=2, quest=False):
    """solving Wahba's problem to find the attitude estimate using quaternions; 
       see pg 147
    :param vset_nrtl: inertial sensor measurement vectors set
    :param vset_body: sensor measurement vectors set in body frame 
    :param weights: weights for each sensor measurement
    :param sensors: number of sensors computed; default=2
    :param quest: boolean to pick between computing eigval/eigvecs
                  or use QUEST method
    :return qset: quaternion set for optimal attitude estimate
                  satisfying Wahba's problem
    """
    B = np.zeros((3,3))
    for s in range(sensors):
        m1 = mat.mxs(scalar=weights[s], \
                          m1=vec.vxvT(v1=vset_body[s], v2=vset_nrtl[s]))
        B = mat.mxadd(m2=m1, m1=B)
    S = mat.mxadd(m2=B, m1=mat.mT(m1=B))
    sigma = B[0][0] + B[1][1] + B[2][2]
    Z = [[B[1][2]-B[2][1]], [B[2][0]-B[0][2]], [B[0][1]-B[1][0]]]
    K = np.zeros((4,4))
    K[0][0] = sigma
    Kterm4 = mat.mxsub(m2=S, m1=mat.mxs(scalar=sigma, m1=np.eye(3)))
    for ii in range(3):
        K[0][ii+1] = Z[ii][0]
        K[ii+1][0] = Z[ii][0]
        for jj in range(3):
            K[ii+1][jj+1] = Kterm4[ii][jj]
    if quest:
        return quest_method(sigma, K, S, Z, weights)
    eigvals, eigvecs = np.linalg.eig(K)
    qset = eigvecs[:, np.argmax(eigvals)]
    dcm = quat.quat2dcm(qset=qset)
    return np.array(qset)


def quest_method(sigma, K, S, Z, weights):
    """solving Wahba's problem using davenport-q method but implementing
    the Quaternion Estimator (QUEST) into finding the optimal lambda (eigval)
    :param sigma: scalar = trace([B])
    :param K: 4x4 matrix
    :param S: 3x3 matrix [S] = [B] + [B].T
    :param Z: 1x3 column vector [Z] = [B23-B32 B31-B13 B12-B21].T
    :param weights: weights for each sensor measurement
    :return crp_opt: optimal CRP based on given parameters
    """
    lambda0 = sum(weights)
    print(f'initial lambda = {lambda0}')
    term1 = np.linalg.inv(mat.mxsub(mat.mxs(lambda0+sigma, np.eye(3,3)), S))
    qvec = mat.mxv(term1, np.vstack(Z))
    print(f'initial crp = {qvec}')

    lam_opt = questf(K, lambda0)
    print(f'optimal lambda = {lam_opt}')

    term1 = np.linalg.inv(mat.mxsub(mat.mxs(lambda0+sigma, np.eye(3,3)), S))
    crp_opt = mat.mxv(term1, np.vstack(Z))
    print(f'optimal CRP = {crp_opt}')

    return np.array(crp_opt)


def questf(K, lambda0):
    """Eigenvalue finder for QUEST method
    :param K: 4x4 matrix
    :param lambda0: initial eigenvalue estimate; sum of the sensor
                    weights
    :return xval: optimal lambda (eigenvalue)
    """
    import scipy.optimize as opt
    lam = lambda0
    for ii in range(4):
        det = qfunc(lam, K)
        print(f'determinant for ({lam}) = {det}')
        xval = opt.newton(qfunc, x0=lam, args=(K,))
        print(f'iteration {ii}: {xval}')
        lam = xval
    return xval


def qfunc(s, K):
    """determinant function for Quaternion Estimator (QUEST) attitude 
    estimator method
    :param s: current lambda value
    :param K: 4x4 matrix
    return: function to compute the QUEST determinant
    """
    return np.linalg.det(mat.mxsub(K, mat.mxs(s, np.eye(4,4))))


def olae_method():
    """Optimal Linear Attitude Estimator (OLAE) deterministic 
    attitude estimation method
    in work
    """
    pass


def wvec_frm_eulerrates_o2b(aset, rates, sequence='321'):
    """orbit to body frame; in work
    """
    s, c = np.sin, np.cos

    if sequence == '321':
        matrix = [[-s(aset[1]),            0.0,        1.0],
                  [s(aset[2])*c(aset[1]),  c(aset[2]), 0.0],
                  [c(aset[2])*c(aset[1]), -s(aset[2]), 0.0]]
    elif sequence == '313':
        matrix = [[s(aset[2])*s(aset[1]),  c(aset[2]),  0.0],
                  [c(aset[2])*s(aset[1]),  -s(aset[2]), 0.0],
                  [c(aset[1]),             0.0,         1.0]]
    else:
        raise ValueError(f'euler sequence not yet implemented')
    return np.array(mat.mxv(m1=matrix, v1=rates))


def wvec_frm_eulerrates_n2b(aset, rates, Omega, sequence='321'):
    """inertial to orbit to body frame; o2 = orbit normal; in work
    """
    if sequence == '321':
        w_o2b = wvec_frm_eulerrates_o2b(aset=aset, rates=rates, sequence='321')
        eulerdcm = euler2dcm(aset, sequence='321')
        w_n2o = vec.vxs(scalar=Omega, v1=mat.mT(eulerdcm)[1])
    else:
        raise ValueError(f'euler sequence not yet implemented')
    return np.array(vec.vxadd(v1=w_o2b, v2=w_n2o))


def eulerrates_frm_wvec_o2b(aset, wvec, sequence='321'):
    """orbit to body frame; in work
    """
    s, c = np.sin, np.cos
    
    if sequence == '321':
        # use the 321 (phi, theta, psi) euler sequence
        matrix = np.array([[0.0,        s(aset[2]),             c(aset[2])],
                          [0.0,        c(aset[2])*c(aset[1]), -s(aset[2])*c(aset[1])],
                          [c(aset[1]), s(aset[2])*s(aset[1]),  c(aset[2])*s(aset[1])]])
        # multiply by the current body angular velocity vector
        matrix = mat.mxs(scalar=1/c(aset[1]), m1=matrix)
    else:
        raise ValueError(f'euler sequence not yet implemented')
    return np.array(mat.mxv(m1=matrix, v1=wvec))


def eulerrates_frm_wvec_n2b(aset, wvec, Omega, sequence='321'):
    """inertial to orbit to body frame; o2 = orbit normal; in work
    """
    if sequence == '321':
        rates_o2b = eulerrates_frm_wvec_o2b(aset=aset, wvec=wvec, sequence='321')
        term2 = vec.vxs(scalar=Omega/np.cos(aset[1]), \
            v1=[np.sin(aset[1])*np.sin(aset[0]), np.cos(aset[1])*np.cos(aset[0]), np.sin(aset[0])])
    else:
        raise ValueError(f'euler sequence not yet implemented')
    return np.array(vec.vxadd(v1=rates_o2b, v2=-term2))


def crprates_frm_wvec_o2b(qset, wvec):
    """orbit to body frame; in work
    """
    q1, q2, q3 = qset

    matrix = np.array([[1+q1**2, q1*q2-q3, q1*q3+q2], 
                      [q2*q1+q3, 1+q2**2, q2*q3-q1],
                      [q3*q1-q2, q3*q2+q1, 1+q3**2]])
    matrix = mat.mxs(0.5, matrix)

    return np.array(mat.mxv(matrix, wvec)) 


def mrprates_frm_wvec_o2b(sigmas, wvec):
    """orbit to body frame; in work
    """
    s1, s2, s3 = sigmas
    s = np.linalg.norm(sigmas)

    matrix = np.array([[1-s**2+2*s1**2, 2*(s1*s2-s3), 2*(s1*s3+s2)], 
                       [2*(s2*s1+s3), 1-s**2+2*s2**2, 2*(s2*s3-s1)],
                       [2*(s3*s1-s2), 2*(s3*s2+s1), 1-s**2+2*s3**2]])
    matrix = mat.mxs(1/4, matrix)

    return np.array(mat.mxv(matrix, wvec))


if __name__ == "__main__":
    
    pass