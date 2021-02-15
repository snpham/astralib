#!/usr/bin/env python3

import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers.constants import *
from traj import conics as con
from math_helpers.vectors import vdotv


def coplanar_transfer(p, e, r1, r2, center='earth'):
    """general form of coplanar circular orbit transfer; an orbit 
    with a flight angle of 0 results in a hohmann transfer;
    :param p: transfer ellipse semi-latus rectum (km)
    :param e: transfer ellipse eccentricity
    :param r1: inner circular orbit radius (km)
    :param r2: outer circular orbit radius (km)
    :param center: planetary center of focus; default=earth
    :return dv1: delta v required to leave inner orbit (km/s)
    :return dv2: delta v required to enter outer orbit (km/s)
    """

    if (p/(1-e)) < r2:
        raise ValueError("Error: transfer orbit apogee is smaller than r2")
    elif (p/(1+e)) > r1:
        raise ValueError("Error: transfer orbit perigee is larger than r1")
    
    mu = get_mu(center=center)

    energy_transfer = -mu*(1-e**2)/(2*p)
    h_transfer = sqrt(mu*p)
    v1_circular = sqrt(mu/r1)
    v1 = sqrt(2*(mu/r1+energy_transfer))
    cos_phi1 = h_transfer/(r1*v1) # angle b/t v1 and v1_circular
    # applying law of cosines to extract 3rd velocity side
    dv1 = sqrt(v1**2+v1_circular**2 - 2*v1*v1_circular*cos_phi1)

    v2 = sqrt(2*(mu/r2+energy_transfer))
    v2_circular = sqrt(mu/r2)
    cos_phi2 = h_transfer/(r2*v2) # angle b/t v1 and v1_circular
    dv2 = sqrt(v2**2+v2_circular**2 - 2*v2*v2_circular*cos_phi2)
    return dv1, dv2


def hohmann_transfer(r1, r2, use_alts=True, get_vtrans=False, center='earth'):
    """hohmann transfer orbit computation from smaller orbit to
    larger; can input either satellite altitude above "object" or
    radius from its center.
    :param r1: altitude (or radius) of smaller circular orbit (km)
    :param r2: altitude (or radius) of larger circular orbit (km)
    :param use_alts: Boolean for switching between r1,r2=altitude 
                     (True) and r1,r2=radius to center
    :param center: planetary center of focus; default=earth
    :return dv1: delta v required to enter transfer orbit (km/s)
    :return dv2: delta v required to enter circular orbit two (km/s)
    """
    # add radius of planet to distance if altitude is inputt
    if use_alts == True and center.lower() == 'earth':
        r1, r2 = [r+r_earth for r in [r1, r2]]

    mu = get_mu(center=center)

    # sma and energy of transfer orbit
    a_trans = (r1+r2)/2
    energy_trans = -mu/(2*a_trans)

    # initial and final velocities
    v_cs1 = sqrt(mu/r1)
    v_cs2 = sqrt(mu/r2)

    # transfer velocities
    v1_trans = sqrt(2*(mu/r1 + energy_trans))
    v2_trans = sqrt(2*(mu/r2 + energy_trans))

    # change and velocities
    dv1 = v1_trans - v_cs1
    dv2 = v_cs2 - v2_trans

    # total deltav and transfer time
    dv_tot = np.abs(dv1) + np.abs(dv2)
    transfer_time = np.pi * sqrt(a_trans**3/mu)

    if get_vtrans:
        return v1_trans, v2_trans, transfer_time

    return dv1, dv2, transfer_time


def bielliptic_transfer(r1, r2, r_trans, use_alts=True, center='earth'):
    """bi-elliptic transfer (hohmann transfer variant) orbit computation 
    from smaller orbit to larger; assumes fpa to be 0
    :param r1: radius of smaller circular orbit (orbit one) (km)
    :param r2: radius of larger circular orbit (orbit two) (km)
    :param r_trans: desired transfer orbit radius (km)
    :param use_alts: Boolean for switching between ri,rf=altitude 
                     (True) and ri,rf=radius to center
    :param center: planetary center of focus; default=earth
    :return dv1: delta v required to enter transfer orbit (km/s)
    :return dv2: delta v required to enter circular orbit two (km/s)
    not tested
    """

    if use_alts == True and center.lower() == 'earth':
        r1, r2, r_trans = [r+r_earth for r in [r1, r2, r_trans]]

    if r_trans < r2:
        raise ValueError("Error: transfer orbit apogee is smaller than r2")

    mu = get_mu(center=center)

    a_trans1 = (r1+r_trans)/2
    a_trans2 = (r2+r_trans)/2
    v_c1 = sqrt(mu/r1) # circular orbit 1
    v_c2 = sqrt(mu/r2) # circular orbit 2
    v_trans1 = sqrt(2*mu/r1 - mu/a_trans1)
    v_transb1 = sqrt(2*mu/r_trans - mu/a_trans1)
    v_transb2 = sqrt(2*mu/r_trans - mu/a_trans2)
    v_trans2 = sqrt(2*mu/r2 - mu/a_trans2)

    dv1 = v_trans1 - v_c1
    dv_trans = v_transb2 - v_transb1
    dv2 = v_c2 - v_trans2
    dv_tot = np.abs(dv1) + np.abs(dv_trans) + np.abs(dv2)

    trans_t = np.pi*sqrt(a_trans1**3/mu) + np.pi*sqrt(a_trans2**3/mu)

    return dv1, dv_trans, dv2, trans_t


def onetangent_transfer(ri, rf, ta_transb, k=0, use_alts=True, center='earth'):
    """Orbit transfer with one tangential burn and one nontangential 
    burn. Must be circular or coaxially elliptic. Currently only for 
    circular orbits.
    :param ri: altitude (or radius) of initial circular orbit (km)
    :param rf: altitude (or radius) of initial circular orbit (km)
    :param ta_transb: true anomaly of transfer orbit at point b (rad)
    :param k: number of revolutions through perigee
    :param use_alts: Boolean for switching between ri,rf=altitude 
                     (True) and ri,rf=radius to center
    :param center: planetary center of focus; default=earth
    :return vtransa: transfer velocity required at point a (km/s)
    :return vtransb: transfer velocity required at point b (km/s)
    :return fpa_transb: flight path angle for the nontangential 
                        transfer (rad)
    :return TOF: time of flight (s)
    in work
    """
    # update constants and parameters
    mu = get_mu(center=center)
    if use_alts and center.lower() == 'earth':
        ri, rf = [r+r_earth for r in [ri, rf]]

    # check location of tangent burn
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

    # compute initial, final, and transfer velocities at a, b
    vi = sqrt(mu/ri)
    vf = sqrt(mu/rf)
    vtransa = sqrt(2*mu/ri - mu/a_trans)
    vtransb = sqrt(2*mu/rf - mu/a_trans)

    # flight path angle of nontangential transfer
    fpa_transb = np.arctan(e_trans*np.sin(ta_transb)
                 / (1+e_trans*np.cos(ta_transb)))

    # get delta-v's at each point and its total
    dva = vtransa - vi
    dvb = sqrt( vtransb**2 + vf**2 - 2*vtransb*vf*np.cos(fpa_transb) )
    dv_otb = np.abs(dva) + np.abs(dvb)

    # computing eccentric anomaly
    E = np.arccos((e_trans+np.cos(ta_transb))/(1+e_trans*np.cos(ta_transb)))

    # computing time of flight
    TOF = sqrt(a_trans**3/mu) * \
        (2*k*np.pi+(E-e_trans*np.sin(E))-(E0 - e_trans*np.sin(E0)))

    return vtransa, vtransb, fpa_transb, TOF


def noncoplanar_transfer(delta, vi, phi_fpa=None, incli=None, inclf=None, change='inc'):
    """noncoplanar transfers to change either inclination only, RAAN 
    only, or both. A dv at nodal points will only change inclination; 
    dv at a certain point in an orbit changes only the RAAN; a dv at 
    any other point changes incl+RAAN
    :param delta: the change value of (incl for incl only; RAAN for 
                  RAAN only or RAAN+incl)
    :param vi: initial velocity at the common point (km/s)
    :param phi_fpa: flight path angle (rad)
    :param incli: inclination of initial orbit (rad)
    :param inclf: inclination of final orbit (rad)
    :param change: choose which element to change;
                   'inc' = inclination only;
                   'raan' = raan only; 'inc+raan' = both elements
    :return dvi: the required delta-velocity for the transfer (km/s)
    :return (arglat1, arglat2): argument of latitude for the common 
                                points, if applicable
    """
    # change inclination only
    if change == 'inc':
        dvi = 2*vi*cos(phi_fpa)*sin(delta/2)
        return dvi
    # change raan only
    elif change == 'raan':
        # NOTE: circular orbits only
        burn_angle = arccos(cos(incli)**2 + sin(incli)**2*cos(delta))
        dvi = 2*vi*sin(burn_angle/2)
        arglat1 = arccos(tan(incli)*(cos(delta)-cos(burn_angle)) / sin(burn_angle))
        arglat2 = arccos(cos(incli)*sin(incli)*(1-cos(delta)) / sin(burn_angle))
        return dvi, (arglat1, arglat2)
    # change both elements
    elif change in ['incl+raan', 'raan+incl', 'inc+raan', 'raan+inc']:
        # NOTE: circular orbits only
        burn_angle = arccos(cos(incli)*cos(inclf)+sin(incli)*sin(inclf)*cos(delta))
        dvi = 2*vi*sin(burn_angle/2)
        arglat1 = arccos((sin(inclf)*cos(delta)-cos(burn_angle)*sin(incli))
                  / (sin(burn_angle)*cos(incli)))
        arglat2 = arccos((cos(incli)*sin(inclf)-sin(incli)*cos(inclf)*cos(delta))
                  / sin(burn_angle))
        return dvi, (arglat1, arglat2)


def combined_planechange(ri, rf, delta_i, use_alts=True, center='earth', get_payload_angle=False):
    """determine the best amount of delta-v to be applied at each node
    for most optimal hohmann transfer with inclination change;
    currently for circular orbits only!
    :param ri: altitude (or radius) of initial circular orbit (km)
    :param rf: altitude (or radius) of initial circular orbit (km)
    :param delta_i: desired inclination change (rad)
    :param use_alts: Boolean for switching between r1,r2=altitude 
                     (True) and ri,rf=radius to center
    :param center: planetary center of focus; default=earth
    :return dva: optimized delta-v at initial node
    :return dvb: optimized delta-v at second node
    :return dii: change in inclination at first node
    :return dif: change in inclination at second node
    """
    
    # get inital parameters
    mu = get_mu(center=center)
    if use_alts == True and center.lower() == 'earth':
        ri, rf = [r+r_earth for r in [ri, rf]]

    # get velocities
    atrans = (ri+rf)/2
    vi = sqrt(mu/ri)
    vf = sqrt(mu/rf)
    vtransa = sqrt(2*mu/ri - mu/atrans)
    vtransb = sqrt(2*mu/rf - mu/atrans)

    # begin intertions
    s = 0.5
    s_prev = 0
    dva = 0
    dvb = 0
    while (np.abs(s-s_prev) > 1e-6):
        s_prev = s
        dva = sqrt(vi**2 + vtransa**2 - 2*vi*vtransa*cos(s*delta_i))
        dvb = sqrt(vf**2 + vtransb**2 - 2*vf*vtransb*cos((1-s)*delta_i))
        s = 1/delta_i * arcsin( dva*vf*vtransb*sin((1-s)*delta_i)
            / (dvb*vi*vtransa) )    

    # get initial and final inclinations
    dii = s*delta_i
    dif = (1-s)*delta_i

    # get optimized delta v's
    dva = sqrt(vi**2 + vtransa**2 - 2*vi*vtransa*cos(dii))
    dvb = sqrt(vf**2 + vtransb**2 - 2*vf*vtransb*cos(dif))

    if get_payload_angle:
        gamma_a = arccos(-(vi**2+dva**2-vtransa**2) / (2*vi*dva))
        gamma_b = arccos(-(vtransb**2+dvb**2-vf**2) / (2*vtransb*dvb))
        return dva, dvb, dii, dif, gamma_a, gamma_b

    return dva, dvb, dii, dif


def patched_conics(r1, r2, rt1, rt2):
    """compute a patched conics orbit transfer from an inner planet to 
    outer planet.
    :param r1: orbital radius around inner planet (km)
    :param r2: orbital radius around outer planet (km)
    :param rt1: radius to inner planet from center of transfer orbit (km)
    :param rt2: radius to outer planet from center of transfer orbi (km)
    :return vt1: departure velocity from inner planet (km/s)
    :return vt2: arrival velocity to outer planer (km/s)
    :return dv_inj: injection velocity to transfer orbit (km/s)
    :return dv_ins: insertion velocity from transfer orbit (km/s)
    :return TOF: transfer time of flight (s)
    only tested for heliocentric earth->mars
    """

    r_orbit1 = r1
    r_orbit2 = r2
    atrans = (rt1 + rt2) / 2 # transfer sma
    TOF = pi*sqrt(atrans**3/mu_sun) # period of hohmann transfer
    # print(f'D: TOF (days): {TOF/(3600*24)}')

    # velocities
    vc1 = sqrt(mu_earth/r_orbit1)
    vc2 = sqrt(mu_mars/r_orbit2)
    vt1 = sqrt(2*mu_sun/rt1 - mu_sun/atrans) # heliocentric, departure
    vt2 = sqrt(2*mu_sun/rt2 - mu_sun/atrans) # heliocentric, arrival
    # print(f'vc1 {vc1}; A1: vt1 {vt1}')
    # print(f'vc2 {vc2}; A2: vt2 {vt2}')

    dv1 = vt1 - vc1
    dv2 = vc2 - vt2
    # print(f'dv1 {dv1}; dv2 {dv2}')

    # velocity of earth and mars rel. to sun
    v_es = sqrt(mu_sun/sma_earth)
    v_ms = sqrt(mu_sun/sma_mars)
    # print(f'v_es {v_es}; v_ms {v_ms}')

    # hyperbolic excess velocity
    v_hyp1 = vt1 - v_es # wrt earth
    v_hyp2 = vt2 - v_ms # wrt mars
    # print(f'v_hyp1 {v_hyp1}; v_hyp2 {v_hyp2}')

    # departure
    vp1 = sqrt(2*mu_earth/r_orbit1 + v_hyp1**2) # earth departure
    # print(f'vp1 {vp1}')
    dv_inj = vp1 - vc1 # v_inf
    # print(f'B: dv_inj {dv_inj}')

    # arrival
    vp2 = sqrt(2*mu_mars/r_orbit2 + v_hyp2**2) # mars arrival
    # print(f'vp2 {vp2}')
    dv_ins = vc2 - vp2
    # print(f'C: dv_ins {dv_ins}')

    # # testing with existing code
    # vsatd, vsata, tt = hohmann_transfer(rt1, rt2, use_alts=False, get_vtrans=True, center='sun')
    # print(vsatd, vsata, tt)

    # print(f"A1 (km/s): {vt1}")
    # print(f"A2 (km/s): {vt2}")
    # print(f"B (km/s): {dv_inj}")
    # print(f"C (km/s): {dv_ins}")
    # print(f"D (days): {TOF/(3600*24)}")

    return vt1, vt2, dv_inj, dv_ins, TOF


def get_c2c3(psi):
    """compute the c2 and c2 coefficients for lamberts algorithm
    :param psi: deltaE*2
    :return c2: c2 coefficient
    :return c3: c3 coefficient
    """
    if psi > 1e-6:
        sqrt_psi = sqrt(psi)
        c2 = ( 1 - cos(sqrt_psi) ) / psi
        c3 = ( sqrt_psi - sin(sqrt_psi) ) / ( sqrt(psi**3) )

    elif psi < -1e-6:
        sqrt_npsi = sqrt(-psi)
        c2 = ( 1 - cosh(sqrt_npsi) ) / psi
        c3 = ( sinh(sqrt_npsi) - sqrt_npsi ) / ( sqrt( (-psi)**3 ) )

    else:
        c2 = 1/2.
        c3 = 1/6.

    return c2, c3


def lambert_univ(ri, rf, TOF0, dm=None, center='sun', 
                 dep_planet=None, arr_planet=None, return_psi=False):
    """lambert solver using universal variables; 0 rev algorithm
    :param ri: position of departure planet at time of departure (km)
    :param rf: position of arrival planet at time of arrival (km)
    :param TOF0: transfer time of flight (s)
    :param dm: direction of motion (optional); if None, then 
               the script will auto-compute direction based on the
               change in true anomaly
    :param center: point where both planets are orbiting about;
                   default = 'sun'
    :return vi: departure velocity of the transfer (km/s)
    :return vf: arrival velocity of the transfer (km/s)
    """

    # throw error if time of flight is outside of feasible values
    if dep_planet in ['mars', 'earth'] and arr_planet in ['mars', 'earth']:
        if TOF0 < 30*24*3600 or TOF0 > 500*24*3600:
            raise ValueError("Earth to Mars TOF out of bounds")

    # set mu = 398600.4418 for matlab vallado test 1
    mu = get_mu(center=center)

    # position vectors and magnitudes
    ri = np.array(ri)
    rf = np.array(rf)
    r0mag = norm(ri)
    rfmag = norm(rf)

    # assuming positions of planets are in the ecliptic
    tanom1 = np.arctan2(ri[1], ri[0])
    tanom2 = np.arctan2(rf[1], rf[0])
    dtanom = tanom2 - tanom1

    if dtanom < 0:
        dtanom += 2*np.pi
    elif dtanom > 2*np.pi:
        dtanom -= 2*np.pi

    # get direction or orbit
    if dm:
        dm = dm
    else:
        if dtanom < np.pi:
            dm = 1
        else:
            dm = -1

    # get constant values of orbit
    cos_dtanom = vdotv(ri, rf) / (r0mag*rfmag)
    A = dm * np.sqrt(r0mag*rfmag*(1+cos_dtanom))

    if dtanom == 0 or A == 0:
        raise ValueError("Trajectory can't be computed")

    # initializing parameters
    psi = 0
    c2 = 1/2
    c3 = 1/6
    psi_up = 4*np.pi**2
    psi_low = -4*np.pi
    TOF = -10.0
    y = 0
    
    while np.abs(TOF - TOF0) > 1e-5:

        y = r0mag + rfmag + A*(psi*c3-1)/sqrt(c2)

        if A > 0 and y < 0:
            while y < 0:
                print('readjusting y')
                N = 0.8
                psi = N*1/c3 * (1-sqrt(c2)/A * (r0mag + rfmag))
                c2, c3 = get_c2c3(psi)
                y = r0mag + rfmag + A*(psi*c3-1)/np.sqrt(c2)

        chi = sqrt(y/c2)
        TOF = (chi**3*c3 + A*sqrt(y)) / sqrt(mu)

        if TOF <= TOF0:
            psi_low = psi
        else:
            psi_up = psi

        psi = (psi_up+psi_low) / 2
        c2, c3 = get_c2c3(psi)

    if return_psi:
        return psi

    # compute f, g functions
    f = 1. - y/r0mag
    gdot = 1. - y/rfmag
    g = A * sqrt(y/mu)

    # velocities
    vi = (rf - f*ri) / g
    vf = (gdot*rf - ri) / g

    return vi, vf


def lambert_univ_multirev(ri, rf, TOF0, nrev=0, ttype=None, dm=None, 
                          psi_min=None, tof_min=None, center='sun', 
                          dep_planet=None, arr_planet=None, return_psi=False):
    """lambert solver using universal variables; n-revs algorithm
    :param ri: position of departure planet at time of departure (km)
    :param rf: position of arrival planet at time of arrival (km)
    :param TOF0: transfer time of flight (s)
    :param nrev: number of revolutions
    :param dm: direction of motion (optional); if None, then 
               the script will auto-compute direction based on the
               change in true anomaly
    :param center: point where both planets are orbiting about;
                   default = 'sun'
    :return vi: departure velocity of the transfer (km/s)
    :return vf: arrival velocity of the transfer (km/s)
    in work
    """

    # set mu = 398600.4418 for matlab vallado test 1
    mu = get_mu(center=center)

    # position vectors and magnitudes
    ri = np.array(ri)
    rf = np.array(rf)
    r0mag = norm(ri)
    rfmag = norm(rf)

    # assuming positions of planets are in the ecliptic
    tanom1 = np.arctan2(ri[1], ri[0])
    tanom2 = np.arctan2(rf[1], rf[0])
    dtanom = tanom2 - tanom1

    if dtanom < 0:
        dtanom += 2*np.pi
    elif dtanom > 2*np.pi:
        dtanom -= 2*np.pi

    # get direction or orbit
    if dm:
        dm = dm
    else:
        if dtanom < np.pi:
            dm = 1
        else:
            dm = -1

    # get constant values of orbit
    cos_dtanom = vdotv(ri, rf) / (r0mag*rfmag)
    A = dm * np.sqrt(r0mag*rfmag*(1+cos_dtanom))

    if dtanom == 0 or A == 0:
        raise ValueError("Trajectory can't be computed")

    # initializing parameters
    psi = 0
    c2 = 1/2
    c3 = 1/6
    if nrev != 0:
        psi_up = 4*(nrev+1)**2*pi**2
        psi_low = 4*nrev**2*pi**2
    else:
        psi_up = 4*np.pi**2
        psi_low = -4*np.pi

    # determine bounds based on Type 3 or 4
    if ttype == 3:
        psi_low = psi_up
        psi_up = psi_min
    elif ttype == 4:
        psi_low = psi_min
        psi_up = psi_up

    # initialize
    TOF = -10.0
    y = 0
    tof_tol = 1e-5
    count = 0
    count2 = 0

    while np.abs(TOF - TOF0) > tof_tol:
    
        y = r0mag + rfmag + A*(psi*c3-1)/sqrt(c2)

        if A > 0 and y < 0:
            while y < 0:
                print('readjusting y')
                N = 0.8
                psi = N*1/c3 * (1-sqrt(c2)/A * (r0mag + rfmag))
                c2, c3 = get_c2c3(psi)
                y = r0mag + rfmag + A*(psi*c3-1)/np.sqrt(c2)

        chi = sqrt(y/c2)
        TOF = (chi**3*c3 + A*sqrt(y)) / sqrt(mu)

        if TOF <= TOF0:
            psi_low = psi
        else:
            psi_up = psi

        psi = (psi_up+psi_low) / 2
        c2, c3 = get_c2c3(psi)



    return psi


def lambert_multrev2(ri, rf, TOF0, dm=None, center='sun', 
                    dep_planet=None, arr_planet=None, return_psi=False,
                    nrev=None, ttype=None, psi_min=None
                    ):
    """lambert solver using universal variables; 0 rev algorithm
    :param ri: position of departure planet at time of departure (km)
    :param rf: position of arrival planet at time of arrival (km)
    :param TOF0: transfer time of flight (s)
    :param dm: direction of motion (optional); if None, then 
               the script will auto-compute direction based on the
               change in true anomaly
    :param center: point where both planets are orbiting about;
                   default = 'sun'
    :return vi: departure velocity of the transfer (km/s)
    :return vf: arrival velocity of the transfer (km/s)
    """

    # throw error if time of flight is outside of feasible values
    if dep_planet in ['mars', 'earth'] and arr_planet in ['mars', 'earth']:
        if TOF0 < 30*24*3600 or TOF0 > 500*24*3600:
            raise ValueError("Earth to Mars TOF out of bounds")

    # set mu = 398600.4418 for matlab vallado test 1
    mu = get_mu(center=center)

    # position vectors and magnitudes
    ri = np.array(ri)
    rf = np.array(rf)
    r0mag = norm(ri)
    rfmag = norm(rf)

    # assuming positions of planets are in the ecliptic
    tanom1 = np.arctan2(ri[1], ri[0])
    tanom2 = np.arctan2(rf[1], rf[0])
    dtanom = tanom2 - tanom1

    if dtanom < 0:
        dtanom += 2*np.pi
    elif dtanom > 2*np.pi:
        dtanom -= 2*np.pi

    # get direction or orbit
    if dm:
        dm = dm
    else:
        if dtanom < np.pi:
            dm = 1
        else:
            dm = -1

    # get constant values of orbit
    cos_dtanom = vdotv(ri, rf) / (r0mag*rfmag)
    A = dm * np.sqrt(r0mag*rfmag*(1+cos_dtanom))

    if dtanom == 0 or A == 0:
        raise ValueError("Trajectory can't be computed")

    # initializing parameters
    psi = 0
    c2 = 1/2
    c3 = 1/6
    TOF = -10.0
    y = 0
    
    if nrev == 1:
        psi_high = 4*(nrev+1)**2*pi**2
        psi_low = 4*nrev**2*pi**2
    else:
        psi_high = 4*np.pi**2
        psi_low = -4*np.pi

    # determine bounds based on Type 3 or 4
    if ttype == 3:
        psi_low = psi_high
        psi_high = psi_min
    elif ttype == 4:
        psi_low = psi_min

    while np.abs(TOF - TOF0) > 1e-5:

        y = r0mag + rfmag + A*(psi*c3-1)/sqrt(c2)

        if A > 0 and y < 0:
            while y < 0:
                print('readjusting y')
                N = 0.8
                psi = N*1/c3 * (1-sqrt(c2)/A * (r0mag + rfmag))
                c2, c3 = get_c2c3(psi)
                y = r0mag + rfmag + A*(psi*c3-1)/np.sqrt(c2)

        chi = sqrt(y/c2)
        TOF = (chi**3*c3 + A*sqrt(y)) / sqrt(mu)

        if ttype == 4:
            if TOF <= TOF0:
                psi_low = psi
            else:
                psi_high = psi
        else:
            if TOF > TOF0:
                psi_low = psi
            else:
                psi_high = psi

        psi = (psi_high+psi_low) / 2
        c2, c3 = get_c2c3(psi)

    if return_psi:
        return psi

    # compute f, g functions
    f = 1. - y/r0mag
    gdot = 1. - y/rfmag
    g = A * sqrt(y/mu)

    # velocities
    vi = (rf - f*ri) / g
    vf = (gdot*rf - ri) / g

    return vi, vf






if __name__ == '__main__':
    
    pass

