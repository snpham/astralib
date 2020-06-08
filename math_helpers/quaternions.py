#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from math_helpers import matrices, rotations, vectors
from numpy.linalg import norm


def qxscalar(scalar, quat):
    """quaternion scalar multiplication
    :param scalar: scalar value
    :param quat: quaternion value
    :return: scaled quaternion set; q_out = scalar * quat
    """
    return [scalar*q for q in quat]


def qadd(quat1, quat2):
    """quaternion addition
    :param quat1: first quaternion
    :param quat2: second quaternion
    :return: sum of the two quaternions; q_out = quat1 + quat2
    """
    return [q1+q2 for q1, q2 in zip(quat1, quat2)]


def qxq(quat1, quat2):
    """computes the quaternion for a composite rotation of two
    quaternion sets.
    :param quat1: left quaternion
    :param quat2: right quaternion
    :return q_out: q_out = quat1*quat2
    """
    s1, b1, b2, b3 = quat2
    matrix = [[ s1, -b1, -b2, -b3],
              [ b1,  s1,  b3, -b2],
              [ b2, -b3,  s1,  b1],
              [ b3,  b2, -b1,  s1]]
    v_out = matrices.mxv(matrix, quat1)
    return v_out


def qxq2(quat1, quat2):
    """computes the quaternion for a composite rotation of two
    quaternion sets.
    :param quat1: left quaternion
    :param quat2: right quaternion
    :return q_out: q_out = quat1*quat2
    """
    p0, p1, p2, p3 = quat1
    q0, q1, q2, q3 = quat2 
    p0q0 = p0*q0
    pdotq = vectors.vdotv(v1=[p1, p2, p3], v2=[q1, q2, q3])
    p0q = vectors.vxscalar(scalar=p0, v1=[q1, q2, q3])
    q0p = vectors.vxscalar(scalar=q0, v1=[p1, p2, p3])
    pcrossr = vectors.vcrossv(v1=[p1, p2, p3], v2=[q1, q2, q3])
    scalar = p0q0 - pdotq
    complexq = [p0q[0]+q0p[0]+pcrossr[0], p0q[1]+q0p[1]+pcrossr[1], p0q[2]+q0p[2]+pcrossr[2]]
    return [scalar, complexq[0], complexq[1], complexq[2]]


def q_operator_vector(quat, v1):
    """used to express a vector in a reference frame as a vector in
    the rotated frame;  equivalent to a rotation of the vector 
    through an angle about the quaternion axis of rotation
    :param quat: quaternion set for a given vector rotation
    :param v1: vector expressed in a reference frame to be rotated
    :return wvec: wvec = qvq*; rotated vector wvec, expressed in the
                  fixed reference frame; 
    """
    q0, q1, q2, q3 = quat
    wvec = np.zeros(3)
    term1 = vectors.vxscalar(scalar=(2.*q0**2-1), v1=v1)
    term2 = vectors.vxscalar(scalar=2.*vectors.vdotv(v1=v1, v2=[q1, q2, q3]), v1=[q1, q2, q3])
    term3 = vectors.vxscalar(scalar=2.*q0, v1=vectors.vcrossv(v1=[q1, q2, q3], v2=v1))
    wvec[0] = sum([term1[0], term2[0], term3[0]])
    wvec[1] = sum([term1[1], term2[1], term3[1]])
    wvec[2] = sum([term1[2], term2[2], term3[2]])
    return wvec


def q_operator_frame(quat, v1):
    """coordinate frame rotation operator; used to express a vector
    in a reference frame as a vector in the rotated frame; 
    :param quat: quaternion set for a given vector rotation
    :param v1: fixed vector expressed in a reference frame
    :return wvec: wvec = q*vq; vector wvec, expressed in the rotated
                  frame;
    """
    q0, q1, q2, q3 = quat
    wvec = np.zeros(3)
    term1 = vectors.vxscalar(scalar=(2.*q0**2-1), v1=v1)
    term2 = vectors.vxscalar(scalar=2.*vectors.vdotv(v1=v1, v2=[q1, q2, q3]), v1=[q1, q2, q3])
    term3 = vectors.vxscalar(scalar=2.*q0, v1=vectors.vcrossv(v1=v1, v2=[q1, q2, q3]))
    wvec[0] = sum([term1[0], term2[0], term3[0]])
    wvec[1] = sum([term1[1], term2[1], term3[1]])
    wvec[2] = sum([term1[2], term2[2], term3[2]])
    return wvec


def qxq_transmute(q1st, q2nd):
    """in work
    """
    s1, b1, b2, b3 = q1st
    matrix = [[ s1, -b1, -b2, -b3],
              [ b1,  s1, -b3,  b2],
              [ b2,  b3,  s1, -b1],
              [ b3, -b2,  b1,  s1]]
    v_out =  matrices.mxv(matrix, q1st)    
    return v_out


def q_conjugate(quat):
    """returns the conjugate of a quaternion
    :param quat: quaternion set
    :return: conjugate of quaternion q_conj = quat*
    in work
    """
    return [quat[0], -quat[1], -quat[2], -quat[3]]


def q_norm(quat1):
    """returns the norm of a quaternion q, (length of q)
    :param quat1: quaternion set
    :return: norm of the quaternion set
    in work
    """
    return np.sqrt(quat1[0]**2 + quat1[1]**2 + quat1[2]**2 + quat1[3]**2)


def prv2dcm(e1, e2, e3, theta):
    """converts a principle rotation vector with an angle to a dcm
    :param e1: first axis-component of principle rotation vector
    :param e2: second axis-component of principle rotation vector
    :param e3: third axis-component of principle rotation vector
    :param theta: magnitude of the angle of rotation, (radians)
    :return dcm: direction cosine matrix representing the fixed-axis
                 rotation.
    in work
    """
    z = 1.0 - np.cos(theta)
    dcm = [[e1*e1*z+np.cos(theta), e1*e2*z+e3*np.sin(theta), e1*e3*z-e2*np.sin(theta)],
           [e2*e1*z-e3*np.sin(theta), e2*e2*z+np.cos(theta), e2*e3*z+e1*np.sin(theta)],
           [e3*e1*z+e2*np.sin(theta), e3*e2*z-e1*np.sin(theta), e3*e3*z+np.cos(theta)]]
    return dcm

    
def quat2dcm(qset):
    """converts a quaternion set into a dcm; performs a coordinate
    transformation of a vector in inertial axes into body axes
    :param qset: unit quaternion set
    :return dcm: direction cosine matrix for a given quaternion set;
                 rotates a vector in inertial axes
    """
    s1, q1, q2, q3 = qset
    dcm = [[s1*s1+q1*q1-q2*q2-q3*q3, 2*(q1*q2+s1*q3), 2*(q1*q3-s1*q2)],
           [2*(q1*q2-s1*q3), s1*s1-q1*q1+q2*q2-q3*q3, 2*(q2*q3+s1*q1)],
           [2*(q1*q3+s1*q2), 2*(q2*q3-s1*q1), s1*s1-q1*q1-q2*q2+q3*q3]]
    return dcm


def dcm2quat(dcm):
    """converts a dcm into a quaternion set; note singularties exist
    when s1=0, use sheppard's method if possible
    :param dcm: direction cosine matrix to extract quaternions from
    :return: quaternion set for a given dcm
    in work
    """
    trace = dcm[0][0] + dcm[1][1] + dcm[2][2]
    s1 = 1./2. * np.sqrt(trace + 1)
    if s1 == 0:
        print("s1=0, singularity exists")
    b1 = (dcm[1][2]-dcm[2][1]) / (4.*s1)
    b2 = (dcm[2][0]-dcm[0][2]) / (4.*s1)
    b3 = (dcm[0][1]-dcm[1][0]) / (4.*s1)
    return [s1, b1, b2, b3]


def dcm2quat_sheppard(dcm):
    """converts a dcm to quaternions based on sheppard's method;
    see pg 110, Analytical Mechanics of Space Systems
    :param dcm: direction cosine matrix to extract quaternions from
    :return: quaternion set for a given dcm
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


def euler2quat(a1st, a2nd, a3rd, sequence='313'):
    """returns quaternion set from an euler sequence
    :param a1st: angle for first axis rotation (rad)
    :param a2nd: angle for second axis rotation (rad)
    :param a3rd: angle for third axis rotation (rad)
    :param sequence: euler rotation sequence
    :return: quaternion set for a given euler rotation
    in work
    """
    s, c = np.sin, np.cos
    a1, a2, a3 = a1st, a2nd, a3rd
    if sequence == '313':
        s1 = c(a2/2.)*c((a3+a1)/2.)
        b1 = s(a2/2.)*c((a3-a1)/2.)
        b2 = s(a2/2.)*s((a3-a1)/2.)
        b3 = c(a2/2.)*s((a3+a1)/2.)
    elif sequence == '321':
        s1 = c(a1/2.)*c(a2/2.)*c(a3/2.)+s(a1/2.)*s(a2/2.)*s(a3/2.)
        b1 = c(a1/2.)*c(a2/2.)*s(a3/2.)-s(a1/2.)*s(a2/2.)*c(a3/2.)
        b2 = c(a1/2.)*s(a2/2.)*c(a3/2.)+s(a1/2.)*c(a2/2.)*s(a3/2.)
        b3 = s(a1/2.)*c(a2/2.)*c(a3/2.)-c(a1/2.)*s(a2/2.)*s(a3/2.)
    return [s1, b1, b2, b3]


def quat2euler(qset, sequence='321'):
    """returns an euler angle set for a given sequence and quaternion
    :param qset: quaternion set to extract euler angles from
    :param sequence: euler angle sequence requested
    :return a1st: angle for first axis rotation (rad)
    :return a2nd: angle for second axis rotation (rad)
    :return a3rd: angle for third axis rotation (rad)
    in work
    """
    dcm = quat2dcm(qset)
    a1st, a2nd, a3rd = rotations.dcm2euler(dcm=dcm, sequence=sequence)
    return [a1st, a2nd, a3rd]


def quat2mrp(qset):
    """computes the modified rodrigues parameters from quaternion sets
    :param qset: euler parameter (quaternion) set b0, b1, b2, b3
    :return: MRP vector (sigma)
    """
    sigma1 = qset[1] / (1.+qset[0])
    sigma2 = qset[2] / (1.+qset[0])
    sigma3 = qset[3] / (1.+qset[0])
    return [sigma1, sigma2, sigma3]


def quat2mrps(qset):
    """computes the shadow set modified rodrigues parameters from
    quaternion sets
    :param qset: euler parameter (quaternion) set b0, b1, b2, b3
    return: shadow set of MRP vector (sigma)
    """
    sigma1 = -qset[1] / (1.-qset[0])
    sigma2 = -qset[2] / (1.-qset[0])
    sigma3 = -qset[3] / (1.-qset[0])
    return [sigma1, sigma2, sigma3]


def mrp2dcm(sigmaset):
    """Transforms a set of modified rodrigues parameters into a dcm
    :param sigmaset: modified rodrigues parameters to convert to dcm
    :return dcm: direction cosine matrix from given MRP
    in work
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


def mrpdot(sigmaset, wvec):
    """compute MRP rates for a given angular velocity
    :param sigmaset: modified rodrigues parameter
    :param wvec: angular velocity (rad/s)
    :return sigmadot: rate of change of the MRP's (rad/s)
    in work
    """
    s = sigmaset
    snorm = np.linalg.norm(s)
    Tsigma = [[1-snorm**2+2*s[0]**2, 2*(s[0]*s[1]-s[2]),   2*(s[0]*s[2]+s[1])],
              [2*(s[1]*s[0]+s[2]),   1-snorm**2+2*s[1]**2, 2*(s[1]*s[2]-s[0])],
              [2*(s[2]*s[0]-s[1]),   2*(s[2]*s[1]+s[0]),   1-snorm**2+2*s[2]**2]]
    Tsigma_scaled = matrices.mxscalar(scalar=1./4., m1=Tsigma)
    sigmadot = matrices.mxv(m1=Tsigma_scaled, v1=wvec)
    return sigmadot


def quat_kde_fromq(qset, wset):
    """in work
    """
    w1, w2, w3 = 0.5 * wset
    matrix = [[0, -w1, -w2, -w3],
              [w1,  0,  w3, -w2],
              [w2,-w3,   0,  w1],
              [w3, w2, -w1,   0]]
    quat_kde = matrices.mxv(matrix, qset)
    return quat_kde


def quat_kde_fromw(qset, wset):
    """in work
    """
    s1, b1, b2, b3 = 0.5 * qset
    matrix = [[s1,-b1,-b2,-b3],
              [b1, s1,-b3, b2],
              [b2, b3, s1,-b1],
              [b3,-b2, b1, s1]]
    wset = [0, wset]
    quat_kde = matrices.mxv(matrix, wset)
    return quat_kde


if __name__ == "__main__":


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

    #testing mrp's
    # qset = [0.961798, -0.14565, 0.202665, 0.112505]
    # sigmaset = quat2mrp(qset)
    # sigmasets = quat2mrps(qset)
    # print(sigmaset, sigmasets)

    # testing quat addition, scalar mult
    # q = [1, 0.2, 0.5, 0.4]
    # scalar = 2
    # q2 = qxscalar(scalar=scalar, quat=q)
    # print(q2)
    # s = qadd(q, q2)
    # print(s)

    # # testing quat products
    # p1 = [3, 1, -2, 1]
    # q1 = [2, -1, 2, 3]
    # result = qxq2(quat1=p1, quat2=q1)
    # print(result)
    # result = qxq(q1st=p1, q2nd=q1)
    # print(result)

    # # quaternion conjugate
    # print(q_conjugate(q1))
    # print(q_norm([2, -1, 2, 3]))

    # p1 = [1, 0, 0, 0]
    # print(qvqt(p1, [1,0,0]))
    frame_T_trn_c2ils= [7.11390093e-06, 8.66068429e-01, 4.12538558e-05, 4.99925468e-01]
    frame_T_hrn_c2ils= [ 2.58902171e-01, -3.42011688e-05,  9.65903548e-01, -2.41405936e-05]
    vec = [0, 0, 1]
    result = q_operator_frame(frame_T_hrn_c2ils, vec)
    print(result)
    result = np.rad2deg(quat2euler(frame_T_hrn_c2ils, sequence='321'))
    print(result)
    dcm = quat2dcm(frame_T_hrn_c2ils)
    axis, angle = rotations.prv_axis(dcm)
    print(axis, np.rad2deg(angle))