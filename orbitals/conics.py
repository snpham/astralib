#!/usr/bin/env python3

import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import vectors as vec
from math_helpers import rotations as rot
from math_helpers import matrices as mat


def get_orbital_elements(rvec, vvec, object='earth'):
    """computes Keplerian elements from positon/velocity vectors
    :param rvec: positional vectors of spacecraft (km)
    :param vvec: velocity vectors of spacecraft (km/s)
    :param center: center object of orbit; default = earth
    :return sma: semi-major axis (km)
    :return e: eccentricity
    :return i: inclination (rad)
    :return raan: right ascending node (rad)
    :return aop: argument of periapsis (rad)
    :return ta: true anomaly (rad)
    """

    # get Keplerian class
    k = Keplerian(rvec, vvec, center=object)

    # K vector and rv inner product
    node_mag = vec.norm(k.node_vec)
    r_dot_v = vec.vdotv(v1=rvec, v2=vvec)

    # eccentricity, semi-major axis, semi-parameter
    e = k.e_mag
    if e != 1:
        zeta = k.v_mag**2/2. - k.mu/k.r_mag
        sma = -k.mu/(2*zeta)
        p = sma*(1-e**2)
    else:
        sma = float('inf')
        p = k.h_mag**2/(k.mu)

    if e == 0:
        lperi = float('nan')
    else:
        lperi = np.arccos(vec.vdotv(k.eccentricity_vector, [1.,0.,0.])/k.e_mag)

    # inclination, true anomaly
    i = k.inclination
    ta = k.true_anomaly

    print(f'Orbital Elements:\n',
          f'Semi-major axis: {sma:0.06f} km\n',
          f'Semi-latus Rectum: {p:0.6f} km\n',
          f'Eccentricity: {e:0.6f}\n',
          f'Inclination: {np.rad2deg(i):0.6f} deg')

    # determine if inclination exists
    if i == 0:
        raan = float('nan')
        aop = float('nan')
        arglat = float('nan')
        tlong = lperi + ta
    else:
        # right ascending node
        raan = k.raan

        # argument of periapsis
        aop = k.aop

        # argument of latitude, true longitude at epoch
        arglat = np.arccos(vec.vdotv(k.node_vec, rvec)/(node_mag*k.r_mag))
        if rvec[2] < 0:
            arglat = np.pi + arglat
        tlong = raan + arglat

    print(f' RAAN: {np.rad2deg(raan):0.6f} deg\n',
            f'Argument of Periapsis: {np.rad2deg(aop):0.6f} deg\n',
            f'Longitude of Periapsis: {np.rad2deg(lperi):0.6f} deg\n',
            f'True Anomaly: {np.rad2deg(ta):0.6f} deg\n',
            f'Argument of Latitude: {np.rad2deg(arglat):0.6f} deg\n',
            f'True longitude at epoch: {np.rad2deg(tlong):0.6f} deg')

    return sma, e, i, raan, aop, ta


def get_rv_frm_elements(p, e, i, raan, aop, ta, object='earth'):

    # determine which planet center to compute
    if object == 'earth':
        mu = 3.986004418e14 * 1e-9 # km^3*s^-2
    
    # declaring trig functions
    s, c = np.sin, np.cos

    # converting elements into state vectors in PQW frame
    r = [p/(1+e*c(ta))*c(ta), p/(1+e*c(ta))*s(ta), 0]
    v = [-np.sqrt(mu/p)*s(ta), np.sqrt(mu/p)*(e+c(ta)), 0]
    
    # get 313 transformation matrix to geocentric-equitorial frame
    m1 = rot.rotate_z(-aop)
    m2 = rot.rotate_x(-i)
    m3 = rot.rotate_z(-raan)
    T_ijk_pqw = mat.mxm(m2=m3, m1=mat.mxm(m2=m2, m1=m1))

    # state vector from PQW to ECI
    r_ijk = mat.mxv(m1=T_ijk_pqw, v1=r)
    v_ijk = mat.mxv(m1=T_ijk_pqw, v1=v)
    return r_ijk, v_ijk


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
    evec_term = vec.vxs(1/e_mag, k.e_vec)
    nvec_term = vec.vxs(np.sqrt(1-(1/e_mag)**2), n_hat)
    S = vec.vxadd(evec_term, nvec_term)
    evec_term = vec.vxs(semi_minor*np.sqrt(1-(1/e_mag)**2), k.e_vec)
    nvec_term = vec.vxs(semi_minor/e_mag, n_hat)
    B = vec.vxadd(evec_term, -nvec_term)

    # T and R vector
    T = vec.vxs(1/np.sqrt(S[0]**2+S[1]**2), [S[1], -S[0], 0.])
    R = vec.vcrossv(v1=S, v2=T)

    # BdotT and BdotR
    B_t = vec.vdotv(v1=B, v2=T)
    B_r = vec.vdotv(v1=B, v2=R)

    # angle between B and T
    theta = np.arccos(B_t/vec.norm(B_t))
    
    return B_t, B_r, theta


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


def lat2rec(lon, lat, elevation, center = 'earth', reference='ellipsoid'):
    """in work
    """
    if center == 'earth' and reference=='ellipsoid':
        # we want the mean sea level "geoid" values
        ra = 6378.136
        rc = 6356.752
        e = 0.08182
        f = (ra-rc)/ra

    theta = theta0 + omega_e*(t-t0) + lambda_e
    x = (ra/np.sqrt(1-e**2*np.sin(lon)**2)+elevation)*np.cos(lon)
    y = (ra*(1-e**2)/np.sqrt(1-e**2*np.sin(lon)**2)+elevation)*np.sin(lon)
    z = lon
    r = [x*np.cos(theta), x*np.sin(theta), z]
    return r


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
    """Class to compute classical Keplerian elements from
    position/velocity vectors. 
    :param rvec: positional vectors of spacecraft (km)
    :param vvec: velocity vectors of spacecraft (km/s)
    :param center: center object of orbit; default = earth
    """

    def __init__(self, rvec, vvec, center='earth'):

        # determine gravitational constant
        if center.lower() == 'earth':
            self.mu = 3.986004418e14 * 1e-9 # km^3*s^-2
        elif center.lower() == 'mars':
            self.mu = 4.282837e13 * 1e-9 # km^3*s^-2

        # position and veloccity
        self.rvec = rvec
        self.vvec = vvec
        self.r_mag = vec.norm(rvec)
        self.v_mag = vec.norm(vvec)

        # angular momentun; orbit normal direction
        self.h_vec = vec.vcrossv(rvec, vvec)
        self.h_mag = vec.norm(self.h_vec)
        self.h_hat = self.h_vec/self.h_mag

        # node vector K
        self.node_vec = vec.vcrossv(v1=[0,0,1], v2=self.h_vec)
        self.node_mag = vec.norm(self.node_vec)
        # eccentricity vector
        self.e_vec = self.eccentricity_vector
        self.e_mag = vec.norm(self.e_vec)
        self.e_hat = self.e_vec/self.e_mag


    @property
    def eccentricity_vector(self):
        """eccentricity vector"""
        scalar1 = self.v_mag**2/self.mu - 1./self.r_mag
        term1 = vec.vxs(scalar=scalar1, v1=self.rvec)
        term2 = -vec.vxs(scalar=vec.vdotv(v1=self.rvec, v2=self.vvec)/self.mu, 
                              v1=self.vvec)
        eccentricity_vec = vec.vxadd(v1=term1, v2=term2) # points to orbit periapsis;
        # e_vec = 0 for circular orbits
        return eccentricity_vec


    @property
    def inclination(self):
        """inclination"""
        return np.arccos(self.h_vec[2]/self.h_mag)


    @property
    def true_anomaly(self):
        """true anomaly"""
        if self.e_mag is 0:
            return float('nan')
        ta = np.arccos(vec.vdotv(self.e_vec, self.rvec)/(self.e_mag*self.r_mag))
        if vec.vdotv(rvec, vvec) < 0:
            ta = 2*np.pi - ta
        return ta


    @property
    def raan(self):
        """right ascending node"""
        if self.inclination is 0:
            return float('nan')
        omega = np.arccos(self.node_vec[0]/self.node_mag)
        if self.node_vec[1] < 0:
            omega = 2*np.pi - omega
        return omega


    @property
    def aop(self):
        """argument_of_periapse"""
        if self.e_mag is 0 or self.node_mag is 0:
            return float('nan')
        argp = np.arccos(vec.vdotv(self.node_vec, self.e_vec)/(self.node_mag*self.e_mag))
        if self.e_vec[2] < 0:
            argp = 2*np.pi - argp
        return argp


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
    rv = get_rv_frm_elements(p=14351, e=0.5, i=np.rad2deg(45), raan=np.rad2deg(30), aop=0, ta=0)
    print(rv) # r=[9567.2, 0], v=[0, 7.9054]

    # testing specifc energy function
    pos = vec.vxs(scalar=1e4, v1=[1.2756, 1.9135, 3.1891]) # km
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


    print('\n\n')

    r = [8773.8938, -11873.3568, -6446.7067]
    v =  [4.717099, 0.714936, 0.388178]
    # elements = Keplerian(r, v)
    elements = get_orbital_elements(r, v)

    sma, e, i, raan, aop, ta = elements
    p = sma*(1-e**2)
    r, v = get_rv_frm_elements(p, e, i, raan, aop, ta)
    print(r, v)

    pos = [6524.834, 6862.875, 6448.296]
    ecf2geo(pos)