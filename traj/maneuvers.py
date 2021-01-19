#!/usr/bin/env python3

import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers.constants import *
from math_helpers import vectors as vec
from math_helpers import rotations as rot
from math_helpers import matrices as mat


def coplanar_transfer(p, e, r1, r2, center='earth'):
    """general form of coplanar circular orbit transfer; an orbit 
    with a flight angle of 0 results in a hohmann transfer;
    :param p: transfer ellipse semi-latus rectum (km)
    :param e: transfer ellipse eccentricity
    :param r1: inner circular orbit radius (km)
    :param r2: outer circular orbit radius (km)
    :param object: planetary object of focus; default = earth
    :return dv1: delta v required to leave inner orbit (km/s)
    :return dv2: delta v required to enter outer orbit (km/s)
    """

    if (p/(1-e)) < r2:
        raise ValueError("Error: transfer orbit apogee is smaller than r2")
    elif (p/(1+e)) > r1:
        raise ValueError("Error: transfer orbit perigee is larger than r1")
    
    mu = get_mu(center=center)

    energy_transfer = -mu*(1-e**2)/(2*p)
    h_transfer = np.sqrt(mu*p)
    v1_circular = np.sqrt(mu/r1)
    v1 = np.sqrt(2*(mu/r1+energy_transfer))
    cos_phi1 = h_transfer/(r1*v1) # angle b/t v1 and v1_circular
    # applying law of cosines to extract 3rd velocity side
    dv1 = np.sqrt(v1**2+v1_circular**2 - 2*v1*v1_circular*cos_phi1)

    v2 = np.sqrt(2*(mu/r2+energy_transfer))
    v2_circular = np.sqrt(mu/r2)
    cos_phi2 = h_transfer/(r2*v2) # angle b/t v1 and v1_circular
    dv2 = np.sqrt(v2**2+v2_circular**2 - 2*v2*v2_circular*cos_phi2)
    return dv1, dv2


def hohmann_transfer(r1, r2, use_alts=True, center='earth'):
    """hohmann transfer orbit computation from smaller orbit to
    larger; can input either satellite altitude above "object" or
    radius from its center.
    :param r1: altitude (or radius) of smaller circular orbit (orbit one) (km)
    :param r2: altitude (or radius) of larger circular orbit (orbit two) (km)
    :param alts: Boolean for switching between r1,r2=altitude (True) and
                 r1,r2=radius to center
    :param object: main planetary object
    :return dv1: delta v required to enter transfer orbit (km/s)
    :return dv2: delta v required to enter circular orbit two (km/s)
    """
    # add radius of planet to distance if altitude is inputt
    if use_alts == True and center.lower() == 'earth':
        r1, r2 = [r+REq_earth for r in [r1, r2]]

    mu = get_mu(center=center)

    # sma and energy of transfer orbit
    a_trans = (r1+r2)/2
    energy_trans = -mu/(2*a_trans)

    # initial and final velocities
    v_cs1 = np.sqrt(mu/r1)
    v_cs2 = np.sqrt(mu/r2)

    # transfer velocities
    v1_trans = np.sqrt(2*(mu/r1 + energy_trans))
    v2_trans = np.sqrt(2*(mu/r2 + energy_trans))

    # change and velocities
    dv1 = v1_trans - v_cs1
    dv2 = v_cs2 - v2_trans

    # total deltav and transfer time
    dv_tot = np.abs(dv1) + np.abs(dv2)
    transfer_time = np.pi * np.sqrt(a_trans**3/mu)

    return dv1, dv2, transfer_time


def bielliptic_transfer(r1, r2, r_trans, use_alts=True, center='earth'):
    """bi-elliptic transfer (hohmann transfer variant) orbit computation 
    from smaller orbit to larger; assumes fpa to be 0
    :param r1: radius of smaller circular orbit (orbit one) (km)
    :param r2: radius of larger circular orbit (orbit two) (km)
    :param rb: 
    :param center: planetary center of smaller orbit
    :return dv1: delta v required to enter transfer orbit (km/s)
    :return dv2: delta v required to enter circular orbit two (km/s)
    not tested
    """

    if use_alts == True and center.lower() == 'earth':
        r1, r2, r_trans = [r+REq_earth for r in [r1, r2, r_trans]]

    if r_trans < r2:
        raise ValueError("Error: transfer orbit apogee is smaller than r2")

    mu = get_mu(center=center)

    a_trans1 = (r1+r_trans)/2
    a_trans2 = (r2+r_trans)/2
    v_c1 = np.sqrt(mu/r1) # circular orbit 1
    v_c2 = np.sqrt(mu/r2) # circular orbit 2
    v_trans1 = np.sqrt(2*mu/r1 - mu/a_trans1)
    v_transb1 = np.sqrt(2*mu/r_trans - mu/a_trans1)
    v_transb2 = np.sqrt(2*mu/r_trans - mu/a_trans2)
    v_trans2 = np.sqrt(2*mu/r2 - mu/a_trans2)

    dv1 = v_trans1 - v_c1
    dv_trans = v_transb2 - v_transb1
    dv2 = v_c2 - v_trans2
    dv_tot = np.abs(dv1) + np.abs(dv_trans) + np.abs(dv2)

    trans_t = np.pi*np.sqrt(a_trans1**3/mu) + np.pi*np.sqrt(a_trans2**3/mu)

    return dv1, dv_trans, dv2, trans_t


if __name__ == '__main__':
    
    #
    alt1 = 7028.137
    alt2 = 42158.137
    # print(hohmann_transfer(alt1, alt2))

    alt1 = 191.34411
    alt2 = 35781.34857
    dv1, dv2, tt = hohmann_transfer(alt1, alt2, use_alts=True, center='earth')
    # print(dv1, dv2, np.abs(dv1)+np.abs(dv2), tt)
    # print(tt/60)

    alt1 = 191.34411
    altb = 503873
    alt2 = 376310
    dv1, dv_trans, dv2, tt = bielliptic_transfer(alt1, alt2, altb, use_alts=True, center='earth')
    # print(dv1, dv_trans, dv2, tt/3600)
    # print(np.abs(dv1)+np.abs(dv2)+np.abs(dv_trans))