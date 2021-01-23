#!/usr/bin/env python3

import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers.constants import *
from math_helpers import vectors as vec
from math_helpers import rotations as rot
from math_helpers import matrices as mat
from traj import conics as con

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


def onetangent_transfer(ri, rf, ta_transb, k=0, use_alts=True, center='earth'):
    """has one tangential burn and one nontangential burn. Must be 
    circular or coaxially elliptic. Currently only for circular
    orbits.
    :param ri:
    :param rf:
    :param vtransb:
    :param k:
    :param center:
    :return:
    in work
    """
    if use_alts and center.lower() == 'earth':
        ri, rf = [r+REq_earth for r in [ri, rf]]

    mu = get_mu(center=center)

    Rinv = ri/rf
    if Rinv > 1: 
        # tangent burn is at apogee
        e_trans = (Rinv-1)/(np.cos(ta_transb)+Rinv)
        a_trans = ri/(1+e_trans)
        E0 = np.pi
    else:
        # tangent burn is at perigee
        e_trans = (Rinv-1)/(np.cos(ta_transb)-Rinv)
        a_trans = ri/(1-e_trans)
        E0 = 0.

    vi = np.sqrt(mu/ri)
    vf = np.sqrt(mu/rf)
    vtransa = np.sqrt(2*mu/ri - mu/a_trans)
    vtransb = np.sqrt(2*mu/rf - mu/a_trans)
    dva = vtransa - vi

    fpa_transb = np.arctan(e_trans*np.sin(ta_transb)/(1+e_trans*np.cos(ta_transb)))

    dvb = np.sqrt( vtransb**2 + vf**2 - 2*vtransb*vf*np.cos(fpa_transb) )

    dv_otb = np.abs(dva) + np.abs(dvb)

    E = np.arccos((e_trans+np.cos(ta_transb))/(1+e_trans*np.cos(ta_transb)))
    TOF = np.sqrt(a_trans**3/mu)*(2*k*np.pi+(E-e_trans*np.sin(E))-(E0 - e_trans*np.sin(E0)))

    # print(Rinv)
    # print(e_trans, a_trans)
    # print(vi, vf, vtransa, vtransb, dva)
    # print(np.rad2deg(fpa_transb))
    # print(dvb, dv_otb)
    # print(TOF/60)

    return vtransa, vtransb, fpa_transb, TOF


def noncoplanar_transfer(delta, phi_fpa, vi, change='inc'):
    """
    """
    if change == 'inc':
        dvi_only = 2*vi*np.cos(phi_fpa)*np.sin(delta/2)

    elif change == 'raan':
        pass
    elif change in ['inc+raan', 'raan+inc']:
        pass

    return dvi_only


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

    alti = 191.34411 # alt, km
    altf = 35781.34857 # km
    ta_trans = np.deg2rad(160)
    vtransa, vtransb, fpa_transb, TOF = onetangent_transfer(alti, altf, ta_trans, k=0, center='earth')


    # circular orbit - incl. change only
    delta = np.deg2rad(15)
    vi = 5.892311
    phi_fpa = 0
    dvi = noncoplanar_transfer(delta, phi_fpa, vi, change='inc')
    # print(dvi) # 1.5382021 km/s

    # elliptical orbit - incl. change only
    delta = np.deg2rad(15)
    e = 0.3
    p = 17858.7836 # km
    argp = np.deg2rad(30)
    tanom = np.deg2rad(330)
    vi = con.vel_mag(e=e, tanom=tanom, p=p)
    phi_fpa = con.flight_path_angle(e, tanom)
    # print(vi) # 1.5382021
    # print(phi_fpa, np.rad2deg(phi_fpa)) # -6.78 deg
    dvi = noncoplanar_transfer(delta, phi_fpa, vi, change='inc')
    # print(dvi) # 1.553727 km/s
    # node check
    tanom = tanom - np.pi
    vi = con.vel_mag(e=e, tanom=tanom, p=p)
    phi_fpa = con.flight_path_angle(e, tanom)
    # print(vi) # 3.568017
    # print(phi_fpa, np.rad2deg(phi_fpa)) # 11.4558 deg
    dvi = noncoplanar_transfer(delta, phi_fpa, vi, change='inc')
    # print(dvi) # 0.912883

