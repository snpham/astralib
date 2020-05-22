#!/usr/bin/env python3

import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import vectors as vec
from math_helpers import rotations as rot
from math_helpers import matrices as mat


def get_orbital_elements(rvec, vvec, object='earth'):

    h_vec = vec.vcrossv(v1=rvec, v2=vvec)
    node_vec = vec.vcrossv(v1=[0,0,1], v2=h_vec)

    r_mag = np.linalg.norm(rvec)
    v_mag = np.linalg.norm(vvec)
    h_mag = np.linalg.norm(h_vec)
    node_mag = np.linalg.norm(node_vec)
    r_dot_v = vec.vdotv(v1=rvec, v2=vvec)
    
    if object == 'earth':
        mu = 3.986004418e14 * 1e-9 # km^3*s^-2
    scalar1 = v_mag**2 - mu/r_mag
    term1 = vec.vxscalar(scalar=scalar1, v1=rvec)
    term2 = vec.vxscalar(scalar=r_dot_v, v1=vvec)
    ecc_vec = vec.vxscalar(scalar=1/mu, v1=vec.vxadd(v1=term1, v2=-term2))

    p = h_mag**2/mu
    e = np.linalg.norm(ecc_vec)
    i = np.arccos(h_vec[2]/h_mag)
    tanom = np.arccos(vec.vdotv(ecc_vec, rvec)/(e*r_mag))
    if vec.vdotv(rvec, vvec) < 0:
        tanom = np.pi + tanom

    print(f'Orbital Elements:\n',
          f'Semi-latus Rectum: {p:0.6f} km\n',
          f'Eccentricity: {e:0.6f}\n',
          f'Inclination: {np.rad2deg(i):0.6f} deg')
    if i == 0:
        raan = 'nan'
        argp = 'nan'
        lperi = np.arccos(vec.vdotv(ecc_vec, [1.0,0.0,0.0])/e)
        arglat = 'nan'
        tlong = lperi + tanom

        print(f' RAAN: Undefined\n',
              f'Argument of Periapsis: Undefined\n',
              f'Longitude of Periapsis: {np.rad2deg(lperi):0.6f} deg\n',
              f'True Anomaly: {np.rad2deg(tanom):0.6f} deg\n',
              f'Argument of Latitude: Undefined\n',
              f'True longitude at epoch: {np.rad2deg(tlong):0.6f} deg')
    else:
        raan = np.arccos(vec.vdotv(node_vec, [1.0, 0.0, 0.0])/node_mag)
        if node_vec[1] < 0:
            raan = np.pi + raan
        argp = np.arccos(vec.vdotv(v1=node_vec, v2=ecc_vec)/(node_mag*e))
        if ecc_vec[2] < 0:
            argp = np.pi + argp
        arglat = np.arccos(vec.vdotv(node_vec, rvec)/(node_mag*r_mag))
        if rvec[2] < 0:
            arglat = np.pi + arglat
        tlong = raan + arglat

        print(f' RAAN: {np.rad2deg(raan):0.6f} deg\n',
              f'Argument of Periapsis: {np.rad2deg(argp):0.6f} deg\n',
              f'True Anomaly: {np.rad2deg(tanom):0.6f} deg\n',
              f'Argument of Latitude: {np.rad2deg(arglat):0.6f} deg\n',
              f'True longitude at epoch: {np.rad2deg(tlong):0.6f} deg')


    return p, e, i, raan, argp, tanom, arglat, tlong


def get_rv_frm_elements(p, e, incl, raan, argp, tanom, object='earth'):
    if object == 'earth':
        mu = 3.986004418e14 * 1e-9 # km^3*s^-2
    s, c = np.sin, np.cos
    r = [p/(1+e*c(tanom))*c(tanom), p/(1+e*c(tanom))*s(tanom)]
    v = [-np.sqrt(mu/p)*s(tanom), np.sqrt(mu/p)*(e+c(tanom))]
    return r, v


if __name__ == "__main__":
    
    from pprint import pprint as pp
    # circular orbit
    r = [12756.2, 0.0, 0.0]
    v = [0.0, 7.90537, 0.0]
    elements = get_orbital_elements(rvec=r, vvec=v)
    print(elements)

    # polar orbit
    r = [8750., 5100., 0.0]
    v = [-3., 5.2, 5.9]
    elements = get_orbital_elements(rvec=r, vvec=v)
    print(elements)

    # polar orbit
    r = [0., -5100., 8750.]
    v = [0., 4.2, 5.9]
    elements = get_orbital_elements(rvec=r, vvec=v)
    print(elements)

    ijkframe = rot.T_pqw2ijk(raan=elements[3], incl=elements[2], argp=elements[4])

    p2j = mat.mxv(m1=ijkframe, v1=[1.0, 0.0, 0.0])
    w2i = mat.mxv(m1=ijkframe, v1=[0.0, 0.0, 1.0])
    q2k = mat.mxv(m1=ijkframe, v1=[0.0, 1.0, 0.0])
    print(p2j, w2i, q2k)
    print(np.linalg.norm(p2j))

    # get r,v from orbital elements
    rv = get_rv_frm_elements(p=14351, e=0.5, incl=np.rad2deg(45), raan=np.rad2deg(30), argp=0, tanom=0)
    print(rv) # r=[9567.2, 0], v=[0, 7.9054]