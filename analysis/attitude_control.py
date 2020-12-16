#!/usr/bin/env python3
import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import quaternions as quat
from math_helpers import matrices as mat
from math_helpers import vectors as vec
import matplotlib.pyplot as plt


"""Gemeral 3-axis attitude control
"""

def fb_gain_selection(I, K, P):
    nat_freq = []
    T_decay = []
    damped_freq = []
    damp_ratio = []
    for nn in range(3):

        nat_freq.append(1/2 * I[nn][nn] * np.sqrt(K * I[nn][nn]))
        T_decay.append(2 * I[nn][nn] / P[nn][nn])
        damped_freq.append(1 / (2*I[nn][nn]) * np.sqrt(K*I[nn][nn] - P[nn][nn]**2))
        damp_ratio.append(P[nn][nn] / np.sqrt(K*I[nn][nn]))

    return nat_freq, T_decay, damped_freq, damp_ratio


def wdot_BN(mrp_BRi, w_BRi, w_BNi, w_RNi, w_RNprev, delta_w_BNi, z_sum):

    # angular acceleration, R/N
    wdot_RNi = (w_RNi - w_RNprev) / dt

    # control; note mrp in control is B/R
    if cc2_5:
        u = - K*mrp_BRi - np.dot(P, w_BRi)
    elif cc2_6:
        u = - K*mrp_BRi \
            - np.dot(P, w_BRi) \
            + np.dot(I, wdot_RNi - vec.vcrossv(w_BNi, w_RNi)) \
            + np.dot(mat.skew(w_BNi), np.dot(I, w_BNi))
    elif cc4_2 or cc4_4:
        u = - K*mrp_BRi \
            - mat.mxv((P + mat.mxm(P, mat.mxs(Ki, I))), delta_w_BNi) \
            - mat.mxv(mat.mxs(K, mat.mxs(Ki, P)), z_sum) \
            + mat.mxv(mat.mxs(Ki, mat.mxm(P, I)), delta_w0_BN) \
            + mat.mxv(I, mat.mxv(dcm_BR, wdot_RNi)) \
            - mat.mxv(I, mat.mxv(mat.skew(w_BNi),  mat.mxv(dcm_BR, w_RNi))) \
            + mat.mxv(mat.skew(w_BNi), mat.mxv(I, w_BNi)) \
            - L
    elif cc5_1:
        u = - K*mrp_BRi \
            - np.dot(P, w_BNi) \
            + np.dot(mat.skew(w_BNi), np.dot(I, w_BNi))
    elif cc6_1:
        u = - K*mrp_BRi \
            - np.dot(P, w_BRi) \
            + np.dot(I, wdot_RNi - vec.vcrossv(w_BNi, w_RNi)) \
            + np.dot(mat.skew(w_BNi), np.dot(I, w_BNi)) \
            - L
        for ii in range(len(u)):
            if np.abs(u[ii]) <= u_max:
                u[ii] = u[ii]
            elif np.abs(u[ii]) > u_max:
                u[ii] = u_max * np.sign(u[ii])

    else: # cc1_4, cc1_5, cc2_7
        u = -K*mrp_BRi \
            - np.dot(P, w_BRi) \
            + np.dot(I, wdot_RNi - vec.vcrossv(w_BNi, w_RNi)) \
            + np.dot(mat.skew(w_BNi), np.dot(I, w_BNi)) \
            - L

    # some unmodeled external torque
    if cc3_4 or cc4_2 or cc4_4:
        dL = np.array([0.5, -0.3, 0.2])
    else:
        dL = np.zeros((3))
    
    # euler's rotational equations of motion, pg 431
    term1 = mat.mxv(mat.skew(w_BNi), mat.mxv(I, w_BNi))
    I_inv = np.linalg.inv(I)
    rhs = -term1 + u + L + dL
    wdot_BNi = mat.mxv(I_inv, rhs)

    u_hist.append(u)

    return wdot_BNi


if __name__ == "__main__":

    # let the reference frame be the same as inertal
    time = 30
    inrtl_ref = False
    cc1_4 = False
    cc1_5 = False
    cc2_5 = False
    cc2_6 = False
    cc2_7 = False
    cc3_4 = False
    cc4_2 = False
    cc4_4 = False
    cc5_1 = False
    cc6_1 = True # 0.5110756 correct

    # set specific time for output
    if cc1_4 or cc3_4:
        inrtl_ref = True
    if cc1_5:
        time = 40
    if cc2_5:
        time = 20
    if cc2_6:
        time = 80
    if cc2_7:
        time = 70
    if cc3_4:
        time = 35
    if cc4_2:
        time = 45
        Ki = 0.005
    if cc4_4:
        time = 35
        Ki = 0
    if cc5_1:
        time = 30
        inrtl_ref = True
    if cc6_1:
        time = 60
        u_max = 1


    # time window
    ti = 0.0
    tf = 4*60
    dt = 0.01
    ets = np.arange(ti, tf, dt)

    # principal inertias and initial state
    I = np.diag([100., 75., 80.]) # kgm2
    mrp_BN0 = np.array([0.1, 0.2, -0.1])
    w_BN0 = np.deg2rad([30., 10., -20.]) # rad/sec

    # gain and control constants
    K = 5.  # Nm
    P = 10. * np.eye(3)  # Nms, [P] = P[eye3x3]
    if cc5_1:
        P = np.diag([22.3607, 19.3649, 20.0])
    f = 0.05 # rad/sec
    L = 0. # external torque
    if cc2_6 or cc2_7:
        L = np.array([0.5, -0.3, 0.2])
    z_integral_sum = 0
    u_hist = []

    # MRP and MRP rates, R/N
    mrp_RN = []
    mrpdot_RN = []
    for et in ets:
        # MRP history - given
        mrp_RN.append([0.2*np.sin(f*et), 
                         0.3*np.cos(f*et), 
                        -0.3*np.sin(f*et)])
        # MRP rate history - given
        mrpdot_RN.append([0.2*f*np.cos(f*et), 
                         -0.3*f*np.sin(f*et), 
                         -0.3*f*np.cos(f*et)])
    mrp_RN = np.array(mrp_RN) # for whole series
    mrpdot_RN = np.array(mrpdot_RN) # for whole series

    # nullify mrp in R/N frame if reference frame = inertial
    if inrtl_ref:
        mrp_RN = 0. * mrp_RN
        mrpdot_RN = 0. * mrpdot_RN

    # compute angular rates, R/N, see pg 128
    w_RN = []
    for mrp, mrpdot in zip(mrp_RN, mrpdot_RN):
        T_RN = quat.mrpdot(mrp, None, return_T_matrix=True)
        w_RN.append(4 * mat.mxv(np.linalg.inv(T_RN), mrpdot))
    w_RN = np.array(w_RN) # for whole series

    # get initial MRP and w, B/R
    mrp_BR0 = quat.mrpxmrp(-mrp_RN[0], mrp_BN0) # FIXME
    w_BR0 = w_BN0 - w_RN[0]
    # print(mrp_RN[0])
    # print(mrp_BN0)
    # print(mrp_BR0)

    # initialize MRP's for B/N and B/R
    mrp_BN = []
    mrp_BR = []
    mrp_BN.append(mrp_BN0) # initial
    mrp_BR.append(mrp_BR0) # initial

    # RN -> BN dcm
    dcm_BR = quat.mrp2dcm2(mrp_BR[0])

    # initialize w's for B/N and B/R
    w_BN = []
    w_BR = []
    w_BN.append(w_BN0) # initial
    w_BR.append(w_BR0) # initial
    w_RN_prev = w_RN[0]

    # delta omega
    delta_w_BN = []
    delta_w0_BN = w_BN0 - mat.mxv(dcm_BR, w_RN_prev)
    delta_w_BN.append(delta_w0_BN)

    for ii, et in enumerate(ets):

        # set current MRPs for each frame
        mrp_BN_i = mrp_BN[ii]
        mrp_RN_i = mrp_RN[ii]
        mrp_BR_i = quat.mrpxmrp(-mrp_RN_i, mrp_BN_i)
        mrp_BR.append(mrp_BR_i)
        # set current angular rates for each frame
        w_BN_i = w_BN[ii]
        w_RN_i = w_RN[ii]
        w_BR_i = w_BN[ii] - w_RN_i

        # delta omega's
        delta_w_BN_i = delta_w_BN[ii]

        # compute new MRP and w
        z_integral_sum += mrp_BR_i * dt
        mrp = mrp_BN_i + quat.mrpdot(mrp_BN_i, w_BN_i) * dt
        w = w_BN_i + wdot_BN(mrp_BR_i, w_BR_i, w_BN_i, w_RN_i, w_RN_prev, delta_w_BN_i, z_integral_sum) * dt

        # update previous w_R/N for next run
        w_RN_prev = w_RN_i

        # switch to shadow set if needed
        if np.linalg.norm(mrp) > 1:
            mrp = quat.mrp_shadow(mrp)

        # append to history list
        mrp_BN.append(mrp)
        
        w_BN.append(w)

        
        try:
            mrp_BR_next = quat.mrpxmrp(-mrp_RN[ii+1],  mrp_BN[ii+1])
            dcm_BR = quat.mrp2dcm(mrp_BR_next)
            delta_w_BN.append(w - mat.mxv(dcm_BR, mrp_RN[ii+1]))
        except IndexError:
            pass

        # output desired MRP's
        if et == time:
            if cc1_4:
                # we want output: mrp_B/N
                print(mrp)
                print(np.linalg.norm(mrp))
            else:
                # we want output: mrp_B/R
                mrpBR = quat.mrpxmrp(-mrp_RN[ii+1], mrp)
                print(np.linalg.norm(mrpBR))

# cc2_6 is 0.1327
# cc2_5 is 0.378
# cc4_2 and cc4_4 is wrong

    nat_freq, T_decay, damped_freq, damp_ratio = fb_gain_selection(I, K, P)
    print(nat_freq, T_decay, damped_freq, damp_ratio)

    mrp_BN = np.vstack(mrp_BN)
    mrp_BN = mrp_BN[1:]
    w_BN = np.vstack(w_BN)
    w_BN = w_BN[1:]
    u_hist = np.vstack(u_hist)
    mrp_BR = np.vstack(mrp_BR)
    mrp_BR = mrp_BR[1:]

    plt.figure(1)
    plt.plot(ets, mrp_BN[:,0])
    plt.plot(ets, mrp_BN[:,1])
    plt.plot(ets, mrp_BN[:,2])
    plt.legend(['mrp1', 'mrp2', 'mrp3'])

    plt.figure(2)
    plt.plot(ets, mrp_BR[:,0])
    plt.plot(ets, mrp_BR[:,1])
    plt.plot(ets, mrp_BR[:,2])
    plt.legend(['mrpBR1', 'mrpBR2', 'mrpBR3'])

    plt.figure(3)
    plt.plot(ets, w_BN[:,0])
    plt.plot(ets, w_BN[:,1])
    plt.plot(ets, w_BN[:,2])
    plt.legend(['w1', 'w2', 'w3'])

    plt.figure(4)
    plt.plot(ets, u_hist[:,0])
    plt.plot(ets, u_hist[:,1])
    plt.plot(ets, u_hist[:,2])
    plt.legend(['u1', 'u2', 'u3'])

    plt.show()