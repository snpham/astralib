#!/usr/bin/env python3
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers.constants import *
from math_helpers.vectors import vdotv


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


def get_psimin(r_dep_planet, r_arr_planet, nrev=0, center='sun'):
    
    # get minimum psi value
    mu = get_mu(center=center)
    rmag_dep = norm(r_dep_planet)
    rmag_arr = norm(r_arr_planet)

    # assuming positions of planets are in the ecliptic
    tanom1 = np.arctan2(r_dep_planet[1], r_dep_planet[0])
    tanom2 = np.arctan2(r_arr_planet[1], r_arr_planet[0])
    dtanom = tanom2 - tanom1

    if dtanom < 0:
        dtanom += 2*np.pi
    elif dtanom > 2*np.pi:
        dtanom -= 2*np.pi

    # get direction or orbit
    dm = None
    if dm:
        dm = dm
    else:
        if dtanom < np.pi:
            dm = 1
        else:
            dm = -1

    cos_dtanom = np.dot(r_dep_planet, r_arr_planet) / (rmag_dep*rmag_arr)
    A = dm * np.sqrt(rmag_dep*rmag_arr*(1+cos_dtanom))
    psi_up = 4*(nrev+1)**2*pi**2
    psi_low = 4*nrev**2*pi**2

    # streamline since we know it's near the center
    # psi_up = psi_low + (psi_up - psi_low)*0.6;             
    # psi_low = psi_low + (psi_up - psi_low)*0.3;             
    # initial estimate, just put in center 
    # psi = (psi_up + psi_low) * 0.5;
    # c2, c3 = get_c2c3(psi)

    TOF_min = 6000*3600*24
    for psi_bound in np.linspace(psi_low, psi_up, 5000):
        # print(psi_bound)
        c2, c3 = get_c2c3(psi_bound)
        y = rmag_dep + rmag_arr + A*(psi_bound*c3-1)/sqrt(c2)
        chi = sqrt(y/c2)
        TOF = (chi**3*c3 + A*sqrt(y)) / sqrt(mu)
        if TOF_min > TOF:
            psi_min = psi_bound
            TOF_min = TOF
    
    return psi_min, TOF_min
    

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
    # if dep_planet in ['earth'] and arr_planet in ['mars']:
    #     if TOF0 < 30*24*3600 or TOF0 > 500*24*3600:
    #         raise ValueError("Earth to Mars TOF out of bounds")

    if TOF0 < 0:
        raise ValueError("Negative time of flight, ending..")


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
    y_prev = -1
    tol = 1e-5
    counter = 0
    
    while np.abs(TOF - TOF0) > tol:

        y = r0mag + rfmag + A*(psi*c3-1)/sqrt(c2)

        if y_prev == y:
            counter += 1
            if counter == 5:
                raise ValueError('failed interation in lambert, ending..')

        if A > 0 and y < 0:
            while y < 0:
                # print('readjusting y')
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

        y_prev = y

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


def lambert_multrev(ri, rf, TOF0, dm=None, center='sun', 
                    dep_planet=None, arr_planet=None, return_psi=False,
                    nrev=None, ttype=None, psi_min=None, compute_psimin=False
                    ):
    """lambert solver using universal variables; nrev algorithm
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
    if dep_planet in ['earth'] and arr_planet in ['mars']:
        if TOF0 < 30*24*3600 or TOF0 > 500*24*3600:
            raise ValueError("Earth to Mars TOF out of bounds")

    if compute_psimin:
        psi_min = get_psimin(ri, rf, nrev=nrev, center=center)[0]


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

    psi_high = 4*(nrev+1)**2*pi**2
    psi_low = 4*nrev**2*pi**2
    # print(psi_low, psi_high)
    # initializing parameters
    c2 = 1/2
    c3 = 1/6
    # psi = psi_min
    TOF = -10.0
    y = 0

    # determine bounds based on Type 3 or 4
    if ttype == 3 or ttype == 5:
        psi_low = psi_low
        psi_high = psi_min
    elif ttype == 4 or ttype == 6:
        psi_low = psi_min
    else:
        print('invalid transfer type')
    # print(psi_low, psi_high)

    # FIXME - need to figure out what's the best psi to use??
    # print(psi_min)
    if ttype == 3 or ttype == 5:
        psi = (psi_high+psi_low) / 2 * 0.9 # 45 works
        # print(psi)
    elif ttype == 4 or ttype == 6:
        psi = (psi_high+psi_low) / 2
        # print(psi)


    # exit()
    # psi = (psi_min + psi_low)/2

    while np.abs(TOF - TOF0) > 1e-4:

        y = r0mag + rfmag + A*(psi*c3-1)/sqrt(c2)

        if A > 0 and y < 0:
            while y < 0:
                # print('readjusting y')
                N = 0.8
                psi = N*1/c3 * (1-sqrt(c2)/A * (r0mag + rfmag))
                c2, c3 = get_c2c3(psi)
                y = r0mag + rfmag + A*(psi*c3-1)/np.sqrt(c2)

        chi = sqrt(y/c2)
        TOF = (chi**3*c3 + A*sqrt(y)) / sqrt(mu)
        # print(y, c2, c3, chi, TOF, TOF0)

        if ttype == 4 or ttype == 6:
            if TOF <= TOF0:
                psi_low = psi
            else:
                psi_high = psi
        elif ttype == 3 or ttype == 5:
            if TOF >= TOF0:
                psi_low = psi
            else:
                psi_high = psi

        psi = (psi_high+psi_low) / 2
        c2, c3 = get_c2c3(psi)

    # print(psi)
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