#!/usr/bin/env python3

import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import vectors as vec
from math_helpers import rotations as rot
from math_helpers import matrices as mat


def get_orbital_elements(rvec, vvec, object='earth'):
    """in work
    """
    k = Keplerian(rvec, vvec, center=object)

    node_vec = vec.vcrossv(v1=[0,0,1], v2=k.h_vec)
    node_mag = vec.norm(node_vec)
    r_dot_v = vec.vdotv(v1=rvec, v2=vvec)


    p = k.h_mag**2/(k.mu)
    e = vec.norm(k.ecc_vec)
    i = k.incl
    tanom = k.true_anom
    if vec.vdotv(rvec, vvec) < 0:
        tanom += np.pi

    print(f'Orbital Elements:\n',
          f'Semi-latus Rectum: {p:0.6f} km\n',
          f'Eccentricity: {e:0.6f}\n',
          f'Inclination: {np.rad2deg(i):0.6f} deg')
    if i == 0:
        raan = 'nan'
        argp = 'nan'
        lperi = np.arccos(vec.vdotv(k.ecc_vec, [1.,0.,0.])/e)
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
        argp = np.arccos(vec.vdotv(v1=node_vec, v2=k.ecc_vec)/(node_mag*e))
        if k.ecc_vec[2] < 0:
            argp = np.pi + argp
        arglat = np.arccos(vec.vdotv(node_vec, rvec)/(node_mag*k.r_mag))
        if rvec[2] < 0:
            arglat = np.pi + arglat
        tlong = raan + arglat

        print(f' RAAN: {np.rad2deg(raan):0.6f} deg\n',
              f'Argument of Periapsis: {np.rad2deg(argp):0.6f} deg\n',
              f'True Anomaly: {np.rad2deg(tanom):0.6f} deg\n',
              f'Argument of Latitude: {np.rad2deg(arglat):0.6f} deg\n',
              f'True longitude at epoch: {np.rad2deg(tlong):0.6f} deg')


    return p, e, i, raan, argp, tanom, arglat, tlong


def bplane_targeting(rvec, vvec, center='earth'):
    """Compute BdotT and BdotR for a given b-plane targeting;
    in work
    """
    k = Keplerian(rvec, vvec, center=center)

    e_mag = vec.norm(k.e_vec)
    if e_mag <= 1:
        raise ValueError(f'e_mag = {e_mag}, non-hyperbolic orbit')

    # unit vector normal to eccentricity vector and orbit normal
    n_hat = vec.vcrossv(k.h_hat, k.e_hat)

    # semiminor axis
    semi_minor = k.h_mag**2/(k.mu*np.sqrt(e_mag**2-1))

    # computing incoming asymptote and B-vector
    evec_term = vec.vxscalar(1/e_mag, k.e_vec)
    nvec_term = vec.vxscalar(np.sqrt(1-(1/e_mag)**2), n_hat)
    S = vec.vxadd(evec_term, nvec_term)
    evec_term = vec.vxscalar(semi_minor*np.sqrt(1-(1/e_mag)**2), k.e_vec)
    nvec_term = vec.vxscalar(semi_minor/e_mag, n_hat)
    B = vec.vxadd(evec_term, -nvec_term)

    # T and R vector
    T = vec.vxscalar(1/np.sqrt(S[0]**2+S[1]**2), [S[1], -S[0], 0.])
    R = vec.vcrossv(v1=S, v2=T)

    # BdotT and BdotR
    B_t = vec.vdotv(v1=B, v2=T)
    B_r = vec.vdotv(v1=B, v2=R)

    # angle between B and T
    theta = np.arccos(B_t/vec.norm(B_t))
    
    return B_t, B_r, theta



def get_rv_frm_elements(p, e, incl, raan, argp, tanom, object='earth'):
    if object == 'earth':
        mu = 3.986004418e14 * 1e-9 # km^3*s^-2
    s, c = np.sin, np.cos
    r = [p/(1+e*c(tanom))*c(tanom), p/(1+e*c(tanom))*s(tanom)]
    v = [-np.sqrt(mu/p)*s(tanom), np.sqrt(mu/p)*(e+c(tanom))]
    return r, v


def T_ijk2topo(lon, lat, frame='sez'):
    if frame == 'sez':
        m1 = rot.rotate_z(lon)
        m2 = rot.rotate_y(lat)
        mat = mat.mxm(m2=m2, m1=m1)
    return mat


def T_pqw2ijk(raan, incl, argp):
    s, c = np.sin, np.cos
    rot_mat = [[c(raan)*c(argp)-s(raan)*s(argp)*c(incl), -c(raan)*s(argp)-s(raan)*c(argp)*c(incl), s(raan)*s(incl)],
               [s(raan)*c(argp)+c(raan)*s(argp)*c(incl), -s(raan)*s(argp)+c(raan)*c(argp)*c(incl), -c(raan)*s(incl)],
               [s(argp)*s(incl), c(argp)*s(incl), c(incl)]]
    return rot_mat


def sp_energy(vel, pos, mu=398600.4418):
    """returns specific mechanical energy (km2/s2), angular momentum
    (km2/s), and flight-path angle (deg) of an orbit; 2body problem
    """
    v_mag =  np.linalg.norm(vel)
    r_mag = np.linalg.norm(pos)
    sp_energy =v_mag**2/2. - mu/r_mag
    ang_mo = vec.vcrossv(v1=pos, v2=vel)
    if np.dot(a=pos, b=vel) > 0:
        phi = np.rad2deg(np.arccos(np.linalg.norm(ang_mo)/(r_mag*v_mag)))
    return sp_energy, ang_mo, phi


def hohmann_transfer(r1, r2, object='earth'):
    """hohmann transfer orbit computation from smaller orbit to
    larger
    :param r1: radius of smaller circular orbit (orbit one) (km)
    :param r2: radius of larger circular orbit (orbit two) (km)
    :param object: planetary object of smaller orbit
    :return dv1: delta v required to enter transfer orbit (km/s)
    :return dv2: delta v required to enter circular orbit two (km/s)
    """
    if object == 'earth':
        mu = 398600.4418
        r1, r2 = [r+6378.137 for r in [r1, r2]]
    a_transfer = (r1+r2)/2
    energy_transfer = -mu/(2*a_transfer)
    v1 = np.sqrt(2*(mu/r1 + energy_transfer))
    v_cs1 = np.sqrt(mu/r1)
    dv1 = v1 - v_cs1
    v2 = np.sqrt(2*(mu/r2 + energy_transfer))
    v_cs2 = np.sqrt(mu/r2)
    dv2 = v_cs2 - v2
    dv_tot = dv1 + dv2
    return dv1, dv2


def coplanar_transfer(p, e, r1, r2, object='earth'):
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

    if object == 'earth':
        mu = 398600.4418
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


class Keplerian(object):
    """Classical Keplerian elements; rvec, vvec must be in km, km/s
    in work
    """

    def __init__(self, rvec, vvec, center='earth'):

        self.rvec = rvec
        self.vvec = vvec
        self.r_mag = vec.norm(rvec)
        self.v_mag = vec.norm(vvec)

        # angular momentun; orbit normal direction
        self.h_vec = vec.vcrossv(rvec, vvec)
        self.h_mag = vec.norm(self.h_vec)
        self.h_hat = self.h_vec/self.h_mag

        if center.lower() == 'earth':
            self.mu = 3.986004418e14 * 1e-9 # km^3*s^-2
        elif center.lower() == 'mars':
            self.mu = 4.282837e13 * 1e-9 # km^3*s^-2

        self.e_vec = self.ecc_vec
        self.e_hat = self.e_vec/vec.norm(self.vvec)

    @property
    def ecc_vec(self):
        """eccentricity vector"""
        scalar1 = self.v_mag**2/self.mu - self.r_mag
        term1 = vec.vxscalar(scalar=scalar1, v1=self.rvec)
        term2 = -vec.vxscalar(scalar=vec.vdotv(v1=self.rvec, v2=self.vvec)/self.mu, 
                              v1=self.vvec)
        eccentricity_vec = vec.vxadd(v1=term1, v2=term2) # points to orbit periapsis
        return eccentricity_vec

    @property
    def incl(self):
        """inclination"""
        return np.arccos(self.h_vec[2]/self.h_mag)

    @property
    def true_anom(self):
        """true anomaly"""
        e = vec.norm(self.e_vec)
        return np.arccos(vec.vdotv(self.e_vec, self.rvec)/(e*self.r_mag))



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

    ijkframe = T_pqw2ijk(raan=elements[3], incl=elements[2], argp=elements[4])

    p2j = mat.mxv(m1=ijkframe, v1=[1.0, 0.0, 0.0])
    w2i = mat.mxv(m1=ijkframe, v1=[0.0, 0.0, 1.0])
    q2k = mat.mxv(m1=ijkframe, v1=[0.0, 1.0, 0.0])
    print(p2j, w2i, q2k)
    print(np.linalg.norm(p2j))

    # get r,v from orbital elements
    rv = get_rv_frm_elements(p=14351, e=0.5, incl=np.rad2deg(45), raan=np.rad2deg(30), argp=0, tanom=0)
    print(rv) # r=[9567.2, 0], v=[0, 7.9054]

    # testing specifc energy function
    pos = vec.vxscalar(scalar=1e4, v1=[1.2756, 1.9135, 3.1891]) # km
    vel = [7.9053, 15.8106, 0.0] # km/s
    sp_energy = sp_energy(vel=vel, pos=pos, mu=398600.4418)
    print(sp_energy)

    #
    r1 = 7028.137
    r2 = 42158.137
    print(hohmann_transfer(r1, r2))

    rvec = [-299761, -440886, -308712]
    vvec = [1.343, 1.899, 1.329]
    bdott, bdotr, theta = bplane_targeting(rvec, vvec, center='mars')
    print(bdott, bdotr, np.rad2deg(theta))