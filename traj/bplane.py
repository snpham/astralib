#!/usr/bin/env python3
import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers.constants import *
from traj import conics
from math_helpers.vectors import vcrossv, vdotv, vxs



def get_rp(vinf, psi, mu):
    rp = mu/vinf**2 * ( 1/(cos((pi-psi)/2)) - 1 )
    return rp


def get_turnangle(vinf, rp, mu):
    psi = pi - 2 * arccos(1 / (1 + vinf**2*rp/mu))
    return psi


def bplane_rv(rvec, vvec, v_inf=None, center='sun'):
    """converts trajectory state vectors to B-Plane parameters for a 
    spacecraft approaching a target body.
    not tested
    """
    mu = get_mu(center=center)
    Kep = conics.Keplerian(rvec, vvec, center=center)

    h_hat = Kep.h_hat
    e_vec = Kep.e_vec
    K_hat = [0, 0, 1]

    # asymptote 1/2 angle
    rho = arccos(1/Kep.e_mag)

    if v_inf:
        S_hat = vxs(1/norm(v_inf), v_inf)
    else:
        S_hat = cos(rho)*Kep.e_hat \
            + sin(rho)*vcrossv(h_hat, e_vec)/norm(vcrossv(h_hat, e_vec))

    T_hat = vcrossv(S_hat, K_hat) / norm(vcrossv(S_hat, K_hat))
    R_hat = vcrossv(S_hat, T_hat)

    a = -mu/(2*norm(v_inf))
    b = abs(a)*sqrt(Kep.e_mag**2 - 1)

    B_hat = vcrossv(S_hat, h_hat)
    B_vec = vxs(b, B_hat)
    BT = vdotv(B_vec, T_hat)
    BR = vdotv(B_vec, R_hat)
    theta = arccos(vdotv(T_hat, B_hat))

    if vdotv(B_hat, R_hat) < 0:
        theta = 2*pi - theta

    return np.array([BT, BR, b, theta])


def bplane_vinf(vinf_in, vinf_out, center='earth'):
    """converts trajectory v-infinity vectors to B-Plane parameters for a 
    spacecraft approaching a flyby.
    :param vinf_in: incoming hyperbolic velocity relative to the center (km/s)
    :param vinf_out: outgoing hyperbolic velocity relative to the center (km/s)
    :param center: planet of targetting b-plane
    :return psi: turn angle (rad)
    :return rp: radius of closest approach (km)
    :return BT: BT vector (km)
    :return BR: BR vector (km)
    :return B: magnitude of B vector (km)
    :return theta: angle between the B and T vector (rad)
    """

    mu = get_mu(center=center)

    vmaginf_in = norm(vinf_in)
    vmaginf_out = norm(vinf_out)
    vinf_cross = vcrossv(vinf_in, vinf_out)
    vinf_dot = vdotv(vinf_in, vinf_out)
    S_hat = vinf_in / vmaginf_in
    h_hat = vinf_cross / norm(vinf_cross)
    B_hat = vcrossv(S_hat, h_hat)
    K_hat = [0, 0, 1]
    T_hat = vcrossv(S_hat, K_hat) / norm(vcrossv(S_hat, K_hat))
    R_hat = vcrossv(S_hat, T_hat)

    psi = arccos(vinf_dot / (vmaginf_in*vmaginf_out))
    rp = mu / vmaginf_in**2 * ( 1/(cos((pi-psi)/2)) - 1 )

    B = mu/vmaginf_in**2 * ( ( 1+vmaginf_in**2*rp/mu )**2 - 1 )**(1/2)
    B_vec = vxs(B, B_hat)
    BT = vdotv(B_vec, T_hat)
    BR = vdotv(B_vec, R_hat)
    theta = arccos(vdotv(T_hat, B_hat))
    if vdotv(B_hat, R_hat) < 0:
        theta = 2*pi - theta

    return np.array([psi, rp, BT, BR, B, theta])