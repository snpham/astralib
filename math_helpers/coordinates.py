#!/usr/bin/env python3

import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import vectors as vec
from math_helpers import rotations as rot
from math_helpers import matrices as mat
from traj import conics


REq_earth = 6378.1363 # (km)
RPolar_earth = 6356.7516005 # (km) 
F_earth = 0.003352813178 # = 1/298.257
E_earth = 0.081819221456
omega_earth = 7.292115e-5 # +/- 1.5e-12 (rad/s)


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


def T_ijk2topo(lon, lat, altitude=0.0, frame='sez', reference='spherical'):
    if frame == 'sez':
        m1 = rot.rotate_z(lon)
        m2 = rot.rotate_y(lat)
        matrix = mat.mxm(m2=m2, m1=m1)
        return matrix


def T_pqw2ijk(raan, incl, argp):
    s, c = np.sin, np.cos
    rot_mat = [[c(raan)*c(argp)-s(raan)*s(argp)*c(incl), -c(raan)*s(argp)-s(raan)*c(argp)*c(incl), s(raan)*s(incl)],
               [s(raan)*c(argp)+c(raan)*s(argp)*c(incl), -s(raan)*s(argp)+c(raan)*c(argp)*c(incl), -c(raan)*s(incl)],
               [s(argp)*s(incl), c(argp)*s(incl), c(incl)]]
    return rot_mat


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
    ijkframe = T_pqw2ijk(raan=elements[3], incl=elements[2], argp=elements[4])
    p2j = mat.mxv(m1=ijkframe, v1=[1.0, 0.0, 0.0])
    w2i = mat.mxv(m1=ijkframe, v1=[0.0, 0.0, 1.0])
    q2k = mat.mxv(m1=ijkframe, v1=[0.0, 1.0, 0.0])
    print(p2j, w2i, q2k)
    print(np.linalg.norm(p2j))
    
    pos = [6524.834, 6862.875, 6448.296]
    ecf2geo(pos)

    lon = np.deg2rad(345. + 35/60. + 51/3600.)
    lat = np.deg2rad(-1 * (7. + 54/60. + 23.886/3600.))
    elev = 56/1000.
    r = lat2rec(lon, lat, elev, latref='geodetic', center='earth', ref='ellipsoid')
    r_truth = [6119.40026932, -1571.47955545, -871.56118090]
    print(r)
    print(r_truth)
    