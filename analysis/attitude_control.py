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

def wdot_BN(mrp_BRi, w_BRi, w_BNi, w_RNi, w_RNprev):

    # angular acceleration, R/N
    wdot_RN = (w_RNi - w_RNprev) / dt

    # control
    if cc2_5:
        u = -K*mrp_BRi - np.dot(P, w_BRi)
    elif cc2_6:
        u = -K*mrp_BRi \
            - np.dot(P, w_BRi) \
            + np.dot(I, wdot_RN - vec.vcrossv(w_BNi, w_RNi)) \
            + np.dot(mat.skew_tilde(w_BNi), np.dot(I, w_BNi))
    else: # cc1_4, cc1_5, cc2_7
        u = -K*mrp_BRi \
            - np.dot(P, w_BRi) \
            + np.dot(I, wdot_RN - vec.vcrossv(w_BNi, w_RNi)) \
            + np.dot(mat.skew_tilde(w_BNi), np.dot(I, w_BNi)) \
            - L

    # some unmodeled external torque
    if cc3_4:
        dL = np.array([0.5, -0.3, 0.2])
    else:
        dL = np.zeros((3))
    
    # euler's rotational equations of motion, pg 431
    term1 = mat.mxv(mat.skew_tilde(w_BNi), mat.mxv(I, w_BNi))
    I_inv = np.linalg.inv(I)
    rhs = -term1 + u + L + dL
    wdot_BN = mat.mxv(I_inv, rhs)

    return wdot_BN


if __name__ == "__main__":

    # let the reference frame be the same as inertal
    time = 30
    inrtl_ref = False
    cc1_4 = False
    cc1_5 = False
    cc2_5 = False
    cc2_6 = False
    cc2_7 = False
    cc3_4 = True

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

    # time window
    ti = 0.0
    tf = 120
    dt = 0.01
    ets = np.arange(ti, tf, dt)

    # principal inertias and initial state
    I = np.diag([100., 75., 80.]) # kgm2
    mrp_BN0 = np.array([0.1, 0.2, -0.1])
    w_BN0 = np.deg2rad([30., 10., -20.]) # rad/sec

    # gain and control constants
    K = 5  # Nm
    P = 10 * np.eye(3)  # Nms, [P] = P[eye3x3]
    f = 0.05 # rad/sec
    L = 0 # external torque
    if cc2_6 or cc2_7:
        L = np.array([0.5, -0.3, 0.2])
    
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
    mrp_BR0 = quat.mrpxmrp(-mrp_RN[0], mrp_BN0)
    w_BR0 = w_BN0 - w_RN[0]

    # initialize MRP's for B/N and B/R
    mrp_BN = []
    mrp_BR = []
    mrp_BN.append(mrp_BN0) # initial
    mrp_BR.append(mrp_BR0) # initial

    # initialize w's for B/N and B/R
    w_BN = []
    w_BR = []
    w_BN.append(w_BN0) # initial
    w_BR.append(w_BR0) # initial
    w_RN_prev = w_RN[0]

    for ii, ti in enumerate(ets):

        # set current MRPs for each frame
        mrp_BN_i = mrp_BN[ii]
        mrp_RN_i = mrp_RN[ii]
        mrp_BR_i = quat.mrpxmrp(-mrp_RN_i, mrp_BN_i)

        # set current angular rates for each frame
        w_BN_i = w_BN[ii]
        w_RN_i = w_RN[ii]
        w_BR_i = w_BN[ii] - w_RN_i

        # compute new MRP and w
        mrp = mrp_BN_i + quat.mrpdot(mrp_BN_i, w_BN_i) * dt
        w = w_BN_i + wdot_BN(mrp_BR_i, w_BR_i, w_BN_i, w_RN_i, w_RN_prev) * dt

        # update previous w_R/N for next run
        w_RN_prev = w_RN_i

        # switch to shadow set if needed
        if np.linalg.norm(mrp) > 1:
            mrp = quat.mrp_shadow(mrp)

        # append to history list
        mrp_BN.append(mrp)
        w_BN.append(w)

        # output desired MRP's
        if ti == time:
            if cc1_4:
                # we want output: mrp_B/N
                print(mrp)
                print(np.linalg.norm(mrp))
            else:
                # we want output: mrp_B/R
                mrp_BR = quat.mrpxmrp(-mrp_RN[ii+1], mrp)
                print(np.linalg.norm(mrp_BR))



    mrp_BN = np.vstack(mrp_BN)
    mrp_BN = mrp_BN[1:]
    w_BN = np.vstack(w_BN)
    w_BN = w_BN[1:]

    plt.figure(1)
    plt.plot(ets, mrp_BN[:,0])
    plt.plot(ets, mrp_BN[:,1])
    plt.plot(ets, mrp_BN[:,2])
    plt.legend(['mrp1', 'mrp2', 'mrp3'])

    plt.figure(2)
    plt.plot(ets, w_BN[:,0])
    plt.plot(ets, w_BN[:,1])
    plt.plot(ets, w_BN[:,2])
    plt.legend(['w1', 'w2', 'w3'])
    plt.show()