#!/usr/bin/env python3
import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import rotations, matrices, quaternions, vectors


def eulerrates_integrator(x0, wt, t):
    """Integrates the euler rate differential kinematic equations
    """
    for ii, et in enumerate(t):

        # get the euler rate
        omega_i = wt[:,ii]
        euler_rate = rotations.eulerrates_frm_wvec_o2b(x0, omega_i)
        x0 = x0 + dt*euler_rate

    return x0


def ep_integrator(x0, wt, t):
    """Integrates the EP differential kinematic equations
    """
    for ii, et in enumerate(t):

        # get the euler rate
        omega_i = wt[:,ii]

        q_rate = quaternions.quat_kde_from_w(x0, omega_i)
        x0 = x0 + dt*q_rate

    return x0


def crp_integrator(x0, wt, t):
    """Integrates the CRP differential kinematic equations
    """
    for ii, et in enumerate(t):

        # get the euler rate
        omega_i = wt[:,ii]

        q_rate = rotations.crprates_frm_wvec_o2b(x0, omega_i)
        x0 = x0 + dt*q_rate

    return x0


def mrp_integrator(x0, wt, t):
    """Integrates the MRP differential kinematic equations;
    see pg. 129 in Analytical Mehanics of Space Systems
    """
    for ii, et in enumerate(t):

        # get the euler rate
        omega_i = wt[:,ii]

        s_rate = rotations.mrprates_frm_wvec_o2b(x0, omega_i)
        x0 = x0 + dt*s_rate

        if np.linalg.norm(x0) > 1:
            x0 = -x0/np.linalg.norm(x0)**2

    return x0





if __name__ == "__main__":

    ############# euler rates integrator
    # time window
    ti = 0.0
    tf = 42.0
    dt = 0.01
    ets = np.arange(ti, tf, dt)

    # initial state
    x0 = np.deg2rad([40,30,80])

    # body angular velocity vector in B frame components
    # 20deg * [yaw, pitch, roll]
    omega = np.deg2rad(20)*np.array([np.sin(0.1*ets), 
                                     0.01*np.ones(len(ets)), 
                                     np.cos(0.1*ets)])

    wt = omega
    euler_output = eulerrates_integrator(x0,wt,ets)
    print(np.linalg.norm(euler_output))
    assert np.allclose(np.linalg.norm(euler_output), 6.120691,  atol=1e-03)

    #######################################


    ############ EP differential kinematic equations
    # time window
    ti = 0.0
    tf = 42.0
    dt = 0.01
    ets = np.arange(ti, tf, dt)

    # initial state
    b0 = np.array([0.408248, 0.0, 0.408248, 0.816497])

    # body angular velocity vector, B frame
    omega = np.deg2rad(20)*np.array([np.sin(0.1*ets), 
                                     0.01*np.ones(len(ets)), 
                                     np.cos(0.1*ets)])
    
    wt = omega
    b_output =  ep_integrator(b0,wt,ets)
    print(np.linalg.norm(b_output[1:4]))
    assert np.allclose(np.linalg.norm(b_output[1:4]), 0.825585)


    #######################################


    ############ CRP differential kinematic equations
    # time window
    ti = 0.0
    tf = 42.0
    dt = 0.01
    ets = np.arange(ti, tf, dt)

    # initial state
    q0 = np.array([0.4, 0.2, -0.1])

    # body angular velocity vector, B frame
    omega = np.deg2rad(3)*np.array([np.sin(0.1*ets), 
                                     0.01*np.ones(len(ets)), 
                                     np.cos(0.1*ets)])
    
    wt = omega
    q_output =  crp_integrator(q0, wt, ets)
    print(np.linalg.norm(q_output))
    assert np.allclose(np.linalg.norm(q_output), 1.199495,  atol=1e-03)


    #######################################


    ############ MRP differential kinematic equations
    # time window
    ti = 0.0
    tf = 42.0
    dt = 0.01
    ets = np.arange(ti, tf, dt)

    # initial state
    sigmas = np.array([0.4, 0.2, -0.1])

    # body angular velocity vector, B frame
    omega = np.deg2rad(20)*np.array([np.sin(0.1*ets), 
                                     0.01*np.ones(len(ets)), 
                                     np.cos(0.1*ets)])
    
    wt = omega
    mrp_output =  mrp_integrator(sigmas, wt, ets)
    print(np.linalg.norm(mrp_output))
    assert np.allclose(np.linalg.norm(mrp_output), 0.639465,  atol=1e-03)

    #######################################


    ############ Gemeral 3-axis attitude control
    # time window
    ti = 0.0
    tf = 41
    dt = 0.01
    ets = np.arange(ti, tf, dt)

    # principal inertias
    inertias = np.diag([100., 75., 80.]) # kgm2
    # print(inertias)
    # initial states
    sigma_BN0 = np.array([0.1, 0.2, -0.1])
    w_BN0 = np.array(np.deg2rad([30., 10., -20.])) # rad/sec

    # gains
    K = 5  # Nm
    P = 10 * np.eye(3)  # Nms, [P] = P[eye3x3]
    f = 0.05 # rad/sec
    # print(P)

    # MRP's
    sigma_RN = []
    for et in ets:
        sigma_RN.append([0.2*np.sin(f*et), 
                         0.3*np.cos(f*et), 
                        -0.3*np.sin(f*et)])
    sigma_RN = np.array(sigma_RN)
    # sigma_RN = np.array([0,0,0])
    sigma_BR = [quaternions.mrpxmrp(sig_RN, sigma_BN0) for sig_RN in sigma_RN]

    # MRP derivative
    sigma_RNdot = np.array([0.2*np.cos(f*ets), 
                           -0.3*np.sin(f*ets), 
                           -0.3*np.cos(f*ets)])
    # sigma_RNdot = np.array([0,0,0])



    # sigma_output =  attitude_integrator(inertias, sigma_BN0, w_BN0, ets, K, P, sigma_RN, sigma_RN_dot)
    # print(np.linalg.norm(mrp_output))
    # assert np.allclose(np.linalg.norm(mrp_output), 0.639465,  atol=1e-03)
