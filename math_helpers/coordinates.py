#!/usr/bin/env python3

import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import vectors as vec
from math_helpers import rotations as rot
from math_helpers import matrices as mat
from traj import conics
from math_helpers.constants import *


def geodet2centric(gd_angle, ref="earth"):
    """convert geodetic latitude to geocentric; earth surface only
    :param gd_angle: geodetic latitude angle (rad)
    :param ref: reference planet to get eccentricity from
    :return: geocentric latitude (rad)
    not tested
    """
    if ref == "earth":
        e = E_earth
    else:
        e = E_earth

    return np.arctan((1-e**2)*np.tan(gd_angle))


def geocentric2det(gc_angle, ref="earth"):
    """convert geocentric latitude to geodetic; earth surface only
    :param gc_angle: geocentric latitude angle (rad)
    :param ref: reference planet to get eccentricity from
    :return: geodetic latitude (rad)
    not tested
    """
    if ref == "earth":
        e = E_earth
    else:
        e = E_earth

    return np.arctan(np.tan(gc_angle)/(1-e**2))


def lat2rec(lon, lat, elev, latref='geodetic', center='earth', ref='ellipsoid'):
    """returns IJK location coordinates on surface of earth
    :param lon: location longitude (rad)
    :param lat: location latitude; geodetic or centric (rad)
    :param elev: location elevation above the ellipsoid
    :return: r vector in rectangular IJK frame (km)
    in work
    """
    if latref == 'geocentric':
        lat = geocentric2det(E_earth, lat)

    C = REq_earth / np.sqrt(1-E_earth**2 * np.sin(lat)**2)
    S = REq_earth*(1-E_earth**2) / np.sqrt(1-E_earth**2*np.sin(lat)**2)

    rx = (C+elev)*np.cos(lat)*np.cos(lon)
    ry = (C+elev)*np.cos(lat)*np.sin(lon)
    rz = (S+elev)*np.sin(lat)

    return np.array([rx, ry, rz])


def pqw2ijk(rvec, vvec, center='earth', output='vector'):
    """transforms from PQW to IJK
    :param rvec: positional vector in PQW frame (km)
    :param vvec: velocity vector in PQW frame (km/s)
    :param center: main body center
    :param output: output either vector or transform matrix
    :return T: PQW to IJK transformation matrix
    :return: r vector in rectangular IJK frame (km)
    not tested
    """
    from traj.conics import Keplerian

    k = Keplerian(rvec, vvec, center)
    rxv = vec.vcrossv(rvec, vvec)

    P = k.e_vec / k.e_mag
    W = rxv / np.linalg.norm(rxv)
    Q = vec.vcrossv(W, P)
    T = np.vstack((P, Q, W)).T

    if output == 'matrix':
        return T
    elif output == 'vector':
        return np.array(mat.mxv(T, rvec))


def ijk2rsw(rvec, vvec, output='vector'):
    """transforms from RSW to IJK
    :param rvec: positional vector in inertial IJK frame (km)
    :param vvec: velocity vector in inertial IJK frame (km/s)
    :return: state vector in rotating RSW frame
    not tested
    """
    rxv = vec.vcrossv(rvec, vvec)

    R = rvec / np.linalg.norm(rvec)
    W = rxv / np.linalg.norm(rxv)
    S = vec.vcrossv(W, R)
    T = np.vstack((R, S, W)).T

    return np.vstack((R, S, W))


def ntw2ijk(rvec, vvec, output='vector'):
    """transforms from NTW to IJK
    :param rvec: positional vector in NTW frame (km)
    :param vvec: velocity vector in NTW frame (km/s)
    :param output: output either vector or transform matrix
    :return T: NTW to IJK transformation matrix
    :return: r vector in rectangular IJK frame (km)
    not tested
    """
    rxv = vec.vcrossv(rvec, vvec)

    Tvec = vvec / np.linalg.norm(vvec)
    W = rxv / np.linalg.norm(rxv)
    N = vec.vcrossv(Tvec, W)
    T = np.vstack((Tvec, N, W)).T

    if output == 'matrix':
        return T
    elif output == 'vector':
        return np.array(mat.mxv(T, rvec))


def sez2ijk(rvec, lat, lst, latref='geodetic', output='vector'):
    """transforms from SEZ to IJK
    :param rvec: postional vector in SEZ frame (km)
    :param lat: latitude (rad)
    :param lst: local sidereal time (rad)
    :param latref: geodetic or geocentric
    :param output: output either vector or transform matrix
    :return T: SEZ to IJK transformation matrix
    :return: r vector in rectangular IJK frame (km)
    not tested
    """
    if latref == 'geocentric':
        lat = geocentric2det(lat, ref="earth")

    rmag = rvec / np.linalg.norm(rvec)
    s, c = np.sin, np.cos
    rsite = np.array([s(lat)*c(lst), c(lat)*s(lst), s(lat)])
    rsite = rmag * rsite

    Z = rsite / np.linalg.norm(rsite)
    kxz = vec.vcrossv(rsite[2], Z)
    E = kxz / np.linalg.norm(kxz)
    S = vec.vcrossv(E, Z)

    T = np.vstack((S, E, Z)).T

    if output == 'matrix':
        return T
    elif output == 'vector':
        return np.array(mat.mxv(T, rvec))


def sez2ecef(rvec, vvec, phi, lam, rsite=None, output='vector'):
    """converts from SEZ to ECEF frame; include rsite in ECEF frame
    to compute r,v from ECEF origin to satellite.
    :param rvec: range vector from a site to satellite (km)
    :param vvec: range rate vector from a site to satellite (km/s)
    :param phi: latitude angle (rad)
    :param lam: azimuth angle (rad)
    :param rsite: optional site positional vector in ECEF frame (km)
    :param output: output either vector or transform matrix
    :return T_ecef: SEZ to ECEF transform matrix
    :return sr_ecef: slant range vector in ECEF frame (km)
    :return srdot_ecef: slant range rate vector in ECEF frame (km/s) 
    :return r_ecef: positional vector in ECEF frame (km)
    :return v_ecef: velocity vector in ECEF frame (km/s) 
    not tested
    """

    T_ecef = mat.mxm(rot.rotate(-lam, 'z'), rot.rotate(-(np.pi/2-phi), 'y'))
    if output == 'vector':
        sr_ecef = mat.mxv(T_ecef, rvec)
        srdot_ecef = mat.mxv(T_ecef, vvec)
        if rsite:
            r_ecef = vec.vxadd(sr_ecef, rsite)
            v_ecef = srdot_ecef
            return r_ecef, v_ecef
        return sr_ecef, srdot_ecef
    else:
        return T_ecef


def pqw2eci_M(raan, incl, argp):
    """return transform matrix from PQW to inertial ECI
    :param raan: right ascending node (rad)
    :param incl: inclination (rad)
    :param argp: argument of perigee (rad)
    :return T: tranformation matrix from PQW to ECI
    not tested
    """
    s, c = np.sin, np.cos
    r, i, a = raan, incl, argp
    T = [[c(r)*c(a)-s(r)*s(a)*c(i), -c(r)*s(a)-s(r)*c(a)*c(i),  s(r)*s(i)],
               [s(r)*c(a)+c(r)*s(a)*c(i), -s(r)*s(a)+c(r)*c(a)*c(i), -c(r)*s(i)],
               [s(a)*s(i),                 c(a)*s(i),                 c(i)]]
    return T


def pqw2eci_M2(raan, incl, argp):
    """return transform matrix from PQW to inertial ECI
    :param raan: right ascending node (rad)
    :param incl: inclination (rad)
    :param argp: argument of perigee (rad)
    :return T: tranformation matrix from PQW to ECI
    not tested
    """
    rot1st = rot.rotate(-argp, axis='z')
    rot2nd  = rot.rotate(-incl, 'x')
    rot3rd = rot.rotate(-raan, 'z')

    return mat.mxm(rot3rd, mat.mxm(rot2nd, rot1st))


def eci2pqw_M(raan, incl, argp):
    """return transform matrix from inertial ECI to PQW
    :param raan: right ascending node (rad)
    :param incl: inclination (rad)
    :param argp: argument of perigee (rad)
    :return T: tranformation matrix from ECI to PQW
    not tested
    """
    rot1st = rot.rotate(raan, axis='z')
    rot2nd  = rot.rotate(incl, 'x')
    rot3rd = rot.rotate(argp, 'z')

    return mat.mxm(rot3rd, mat.mxm(rot2nd, rot1st))


def ecef2lat(rvec):
    """determine the geocentric/geodetic latitude, longitude, and height of
    a satellite above a reference ellipsoid.
    :param rvec: positional vector in geocentric equatorial system (ECEF) (km)
    :return lat: latitudinal coordinate (rad)
    :return lon: longitudinal coordinate (rad)
    :return h_ellp: height above reference ellipsoid (km)
    in work
    """
    # equatorial pojection of satellite's position vector
    r_del_sat = np.linalg.norm([rvec[0], rvec[1]])

    # right ascension
    alpha = np.arcsin(rvec[1]/r_del_sat)
    delta = np.arcsin(rvec[2]/np.linalg.norm(rvec))
    lon = alpha

    # iteration
    phi_gd = delta
    phi_gd_prev = 0
    C = 0

    tol = 1e-12
    while (phi_gd - phi_gd_prev > tol):

        phi_gd_prev = phi_gd
        C = REq_earth / np.sqrt( 1 - E_earth**2 * np.sin(phi_gd)**2 )
        phi_gd = np.arctan( (rvec[2] + C*E_earth**2*np.sin(phi_gd)) / r_del_sat )

        print(C, np.rad2deg(np.tan(phi_gd)))

    h_ellp = r_del_sat / np.cos(phi_gd) - C

    lat = phi_gd
    return lat, lon, h_ellp


def ijk2topo(lon, lat, altitude=0.0, frame='sez', reference='spherical'):
    if frame == 'sez':
        m1 = rot.rotate(lon, axis='z')
        m2 = rot.rotate(lat, axis='y')
        matrix = mat.mxm(m2=m2, m1=m1)
        return matrix


def ecf2geo(pos):
    """convert earth-centered fixed to latitude/longitude
    reference: Astronomical Almanac
    in work. see pg 172 in Fund of Astro
    """
    r_earth = 6378.1363
    e_earth = 0.081819221456
    r_dsat = np.sqrt(pos[0]**2+pos[1]**2)
    alpha = np.arcsin(pos[1]/r_dsat)
    lambd = alpha
    print(f'Longitude: {np.rad2deg(lambd)} deg')
    delta = np.arcsin(pos[2]/np.linalg.norm(pos))
    phi_gd = delta
    r_delta = r_dsat
    r_k = pos[2]

    prev_phi = 0
    while (phi_gd - prev_phi) > 0.0000001:
        C = r_earth / np.sqrt(1-e_earth**2*np.sin(phi_gd)**2)
        prev_phi = phi_gd
        phi_gd = np.arctan2((pos[2]+C*e_earth**2*np.sin(phi_gd)),r_delta)
        print(f'Latitude: {np.rad2deg(phi_gd)} deg')

    h_ellp = r_delta/np.cos(phi_gd) - C
    print(f'Altitude: {h_ellp} km')


if __name__ == '__main__':

    # polar orbit
    r = [0., -5100., 8750.]
    v = [0., 4.2, 5.9]
    elements = conics.get_orbital_elements(rvec=r, vvec=v)
    print(elements)
    ijkframe = pqw2eci_M(raan=elements[3], incl=elements[2], argp=elements[4])
    p2j = mat.mxv(m1=ijkframe, v1=[1.0, 0.0, 0.0])
    w2i = mat.mxv(m1=ijkframe, v1=[0.0, 0.0, 1.0])
    q2k = mat.mxv(m1=ijkframe, v1=[0.0, 1.0, 0.0])
    print(p2j, w2i, q2k)
    print(np.linalg.norm(p2j))
    
    pos = [6524.834, 6862.875, 6448.296]
    ecf2geo(pos)

    rvec = [0., -5100., 8750.]
    vvec = [0., 4.2, 5.9]
    r_IJK = pqw2ijk(rvec, vvec)
    print(r_IJK, '\n')


    r_ecef = [6524.834, 6862.875, 6448.296]
    lat, lon, h = ecef2lat(r_ecef)
    print( np.rad2deg(lat), np.rad2deg(lon), h)