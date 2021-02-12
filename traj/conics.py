#!/usr/bin/env python3
import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import vectors as vec
from math_helpers import rotations as rot
from math_helpers import matrices as mat
from math_helpers.constants import *


def get_orbital_elements(rvec, vvec, center='earth'):
    """computes Keplerian elements from positon/velocity vectors
    :param rvec: positional vectors of spacecraft [IJK?] (km)
    :param vvec: velocity vectors of spacecraft [IJK?] (km/s)
    :param center: center object of orbit; default=earth
    :return sma: semi-major axis (km)
    :return e: eccentricity
    :return i: inclination (rad)
    :return raan: right ascending node (rad)
    :return aop: argument of periapsis (rad)
    :return ta: true anomaly (rad)
    """

    # get Keplerian class
    k = Keplerian(rvec, vvec, center=center)

    # eccentricity, specific energy, semi-major axis, semi-parameter
    e = k.e_mag
    zeta = k.v_mag**2/2. - k.mu/k.r_mag
    if e != 1:
        sma = -k.mu/(2*zeta)
        p = sma*(1-e**2)
    else:
        sma = float('inf')
        p = k.h_mag**2/(k.mu)

    # node vec, inclination, true anomaly, mean/eccentric anomaly
    node_vec = k.node_vec
    node_mag = vec.norm(node_vec)
    i = k.inclination
    ta = k.true_anomaly
    if e < 1:
        M, E = mean_anomalies(e, ta)
    elif e == 1:
        M = mean_anomalies(e, ta) # parabolic anomaly
    else:
        M = mean_anomalies(e, ta) # hyperbolic anomaly

    # true longitude of periapsis - from vernal equinox to eccentricty vector
    if e == 0:
        true_lon_peri = float('nan')
    else:
        true_lon_peri = arccos(vec.vdotv(k.eccentricity_vector, [1.,0.,0.])/k.e_mag)
        if k.e_vec[1] < 0:
            true_lon_peri = 2*np.pi - true_lon_peri
    lon_peri_mean = k.raan + k.aop  # for small inclinations

    # RAAN, argument of periapsis, arg. of lat., true long.
    # for inclined orbits
    if i != 0:
        raan = k.raan
        aop = k.aop

        # argument of latitude - from ascending node to satellite position
        # vector in direction of satellite motion
        arglat = arccos(vec.vdotv(node_vec, rvec)/(node_mag*k.r_mag))
        if rvec[2] < 0:
            arglat = 2*np.pi - arglat
        if e != 0:
            # can also use arglat = aop + ta for inclined elliptical orbits
            arglat = aop + ta
            arglat_mean = aop + M # mean = includes mean anomaly
        else:
            arglat_mean = arglat
        true_lon = raan + aop + ta
    # for equatorial orbits
    else:
        raan = float('nan')
        aop = float('nan')
        arglat = float('nan')
        arglat_mean = float('nan')
        true_lon = float('nan')

        # for circular and equatorial orbits
        # true longitude - from vernal equinox to satellite position
        if e == 0:
            true_lon = arccos(vec.vdotv([1.,0.,0.], rvec)/k.r_mag)
            if rvec[1] < 0:
                true_lon = 2*np.pi - true_lon
    mean_lon = true_lon_peri + M  # for small incl and e

    print(f'\nOrbital Elements:\n',
          f'Semi-major axis: {sma:0.06f} km\n',
          f'Semi-latus Rectum: {p:0.6f} km\n',
          f'Eccentricity: {e:0.6f}\n',
          f'Inclination: {np.rad2deg(i):0.6f} deg')

    print(f' RAAN: {np.rad2deg(raan):0.6f} deg\n',
          f'Argument of Periapsis: {np.rad2deg(aop):0.6f} deg\n',
          f'True Longitude of Periapsis: {np.rad2deg(true_lon_peri):0.6f} deg\n',
          f'Mean Longitude of Periapsis: {np.rad2deg(lon_peri_mean):0.6f} deg\n',
          f'True Anomaly: {np.rad2deg(ta):0.6f} deg\n',
          f'Argument of Latitude: {np.rad2deg(arglat):0.6f} deg\n',
          f'Argument of Latitude - Mean: {np.rad2deg(arglat_mean):0.6f} deg\n',
          f'True longitude: {np.rad2deg(true_lon):0.6f} deg\n',
          f'Mean Longitude: {np.rad2deg(mean_lon):0.6f} deg')

    return np.array([sma, e, i, raan, aop, ta])


def get_rv_frm_elements(p, e, i, raan, aop, ta, center='earth'):
    """computes positon/velocity vectors from Keplerian elements.
    We first compute pos/vel in the PQW system, then rotate to the
    geocentric equatorial system.
    :param center: center object of orbit; default=earth
    :param sma: semi-major axis (km)
    :param e: eccentricity
    :param i: inclination (rad)
    :param raan: right ascending node (rad)
    :param aop: argument of periapsis (rad)
    :param ta: true anomaly (rad)
    :return rvec: positional vectors of spacecraft [IJK] (km)
    :return vvec: velocity vectors of spacecraft [IJK] (km/s)
    """
    # determine which planet center to compute from
    mu = get_mu(center=center)

    # declaring trig functions
    s, c = sin, cos

    # assigning temporary variables
    aop_t = aop
    raan_t = raan
    ta_t = ta 

    # checking for undefined states 
    if e == 0 and i == 0:
        aop_t = 0.
        raan_t = 0.
        ta_t = aop_t + raan_t + ta
    elif e == 0:
        aop_t = 0.
        ta_t = aop_t + ta
    elif i == 0:
        raan_t = 0.
        aop_t = raan_t + aop
        ta_t = ta

    # converting elements into state vectors in PQW frame
    r_pqw = [p*c(ta_t) / (1+e*c(ta_t)), p*s(ta_t) / (1+e*c(ta_t)), 0]
    v_pqw = [-sqrt(mu/p)*s(ta_t), sqrt(mu/p)*(e+c(ta_t)), 0]
    
    # get 313 transformation matrix to geocentric-equitorial frame
    m1 = rot.rotate(-aop, axis='z')
    m2 = rot.rotate(-i, axis='x')
    m3 = rot.rotate(-raan, axis='z')
    T_ijk_pqw = mat.mxm(m2=m3, m1=mat.mxm(m2=m2, m1=m1))

    # state vector from PQW to ECI
    r_ijk = mat.mxv(m1=T_ijk_pqw, v1=r_pqw)
    v_ijk = mat.mxv(m1=T_ijk_pqw, v1=v_pqw)
    return np.array(r_ijk), np.array(v_ijk)


def get_rv_frm_elements2(elements, center='earth'):
    """computes positon/velocity vectors from Keplerian elements.
    We first compute pos/vel in the PQW system, then rotate to the
    geocentric equatorial system.
    :param center: center object of orbit; default=earth
    :param sma: semi-major axis (km)
    :param e: eccentricity
    :param i: inclination (rad)
    :param Om: right ascending node (rad)
    :param w: argument of periapsis (rad)
    :param ta: true anomaly (rad)
    :return rvec: positional vectors of spacecraft [IJK] (km)
    :return vvec: velocity vectors of spacecraft [IJK] (km/s)
    output similar answers but not completely tested
    """

    a, e, i, Om, w, ta = elements

    p = a*(1-e**2)

    mu = get_mu(center=center)
    r = p / (1+e*cos(ta))
    h = sqrt( mu*a*(1-e**2) )

    rvec = [r*(cos(Om)*cos(w+ta) - sin(Om)*sin(w+ta)*cos(i)), 
         r*(sin(Om)*cos(w+ta) + cos(Om)*sin(w+ta)*cos(i)), 
         r*(sin(i)*sin(w+ta))
    ]

    vvec = [rvec[0]*h*e*sin(ta)/(r*p) - h/r*(cos(Om)*sin(w+ta) + sin(Om)*cos(w+ta)*cos(i)),
         rvec[1]*h*e*sin(ta)/(r*p) - h/r*(sin(Om)*sin(w+ta) - cos(Om)*cos(w+ta)*cos(i)),
         rvec[2]*h*e*sin(ta)/(r*p) + h/r*(sin(i)*cos(w+ta))
    ]

    return np.hstack([rvec, vvec])


def kepler_prop(r, v, dt, center='earth'):
    """Solve Kepler's problem using classical orbital elements; no perturbations
    :param r: initial position
    :param v: initial velocity
    :param dt: time of flight
    :return rvec: propagated position vector
    :return vvec: propagated velocity vector
    in work
    """
    # determine which planet center to compute from
    mu = get_mu(center=center)

    elements = get_orbital_elements(r, v)
    sma, e, i, raan, aop, ta = elements

    p = sma*(1-e**2)
    n = 2*sqrt(mu/p**3)
    rmag = norm(r)

    if e != 0:
        if e < 1:
            E0, _ = mean_anomalies(e, ta)
            M0 = E0 - e*sin(E0)
            M = M0 + n*dt
            E = univ_anomalies(M=M, e=e, dt=None, p=None)
            ta = true_anomaly(e, p=None, r=None, E=E, B=None, H=None)
        elif e == 1:
            B0 = mean_anomalies(e, ta)
            hvec = vec.vcrossv(r, v)
            hmag = norm(hvec)
            p = hmag**2/mu
            M0 = B0 + B0**3/3
            B = univ_anomalies(M=None, e=e, dt=dt, p=p)
            ta = true_anomaly(e, p=p, r=rmag, E=None, B=B, H=None)
        elif e > 1:
            H0 = mean_anomalies(e, ta)
            M0 = e*sinh(H0) - H0
            M = M0 + n*dt
            H = univ_anomalies(M=M, e=e, dt=None, p=None)
            ta = true_anomaly(e, p=None, r=None, E=None, B=None, H=H)
    else:
        E = raan + aop + ta

    rvec, vvec = get_rv_frm_elements(p, e, i, raan, aop, ta)
    return np.array(rvec), np.array(vvec)



def flight_path_angle(e, ta):
    """computes flight path angle for a satellite; measured from the
    local horizon to the velocity vector
    :param e: magnitude of eccentricity vector
    :param ta: true anomaly (rad)
    :return: flight path angle (rad)
    not tested
    """
    if e == 0:
        return 0.
    elif e < 1:
        E, _ = mean_anomalies(e, ta)
        # FIXME: is this the correct way to check sign?
        fpa = arccos(sqrt((1-e**2)/(1-e**2*cos(E)**2)))
        if ta > np.pi or ta < 0:
            fpa = -fpa
        # ALT: fpa = arctan2(e*sin(ta), 1+e*cos(ta))
    elif e == 1:
        fpa = ta/2.
    else: # if e > 1
        H = mean_anomalies(e, ta)
        fpa = arccos(sqrt( (e**2-1)/(e**2*cosh(H)**2-1) ))
    # return arccos( (1+e*cos(ta) / (sqrt(1+2*e*cos(ta)+e**2))))
    return fpa
    

def bplane_targeting(rvec, vvec, center='earth'):
    """Compute BdotT and BdotR for a given b-plane targeting;
    in work
    """
    k = Keplerian(rvec, vvec, center=center)

    e_mag = vec.norm(k.e_vec)
    if e_mag <= 1:
        raise ValueError(f'e_mag = {e_mag}, non-hyperbolic orbit')

    # unit vector normal to eccentricity vector and orbit normal
    n_hat = vec.vcrossv(k.h_vec/k.h_mag, k.e_vec/k.e_mag)

    # semiminor axis
    semi_minor = k.h_mag**2/(k.mu*sqrt(e_mag**2-1))

    # computing incoming asymptote and B-vector
    evec_term = vec.vxs(1/e_mag, k.e_vec)
    nvec_term = vec.vxs(sqrt(1-(1/e_mag)**2), n_hat)
    S = vec.vxadd(evec_term, nvec_term)
    evec_term = vec.vxs(semi_minor*sqrt(1-(1/e_mag)**2), k.e_vec)
    nvec_term = vec.vxs(semi_minor/e_mag, n_hat)
    B = vec.vxadd(evec_term, -nvec_term)

    # T and R vector
    T = vec.vxs(1/sqrt(S[0]**2+S[1]**2), [S[1], -S[0], 0.])
    R = vec.vcrossv(v1=S, v2=T)

    # BdotT and BdotR
    B_t = vec.vdotv(v1=B, v2=T)
    B_r = vec.vdotv(v1=B, v2=R)

    # angle between B and T
    theta = arccos(B_t/vec.norm(B_t))
    
    return B_t, B_r, theta


def sp_energy(vel, pos, mu=mu_earth):
    """returns specific mechanical energy (km2/s2), angular momentum
    (km2/s), and flight-path angle (deg) of an orbit; 2body problem
    """
    v_mag =  norm(vel)
    r_mag = norm(pos)
    energy =v_mag**2/2. - mu/r_mag
    ang_mo = vec.vcrossv(v1=pos, v2=vel)
    if np.dot(a=pos, b=vel) > 0:
        phi = np.rad2deg(arccos(norm(ang_mo)/(r_mag*v_mag)))
    else:
        phi = 0.
    return energy, ang_mo, phi


def mean_anomalies(e, ta):
    """in work
    """
    if e < 1:
        E = arcsin(sin(ta)*sqrt(1-e**2) / (1+e*cos(ta)))
        # alt: E = arccos((e+cos(ta))/(1+e*cos(ta)))
        M = E - e**sin(E)
        return E, M
    elif e == 1:
        B = tan(ta/2)
        return B
    elif e > 1:
        H = arcsinh( (sin(ta)*sqrt(e**2-1)) / (1+e*cos(ta)) )
        # alt: H = arccosh((e+cos(ta)/(1+e*cos(ta)))
        return H


def true_anomaly(e, p=None, r=None, E=None, B=None, H=None):
    """in work
    """
    if e < 1 and E:
        ta = arcsin(sin(E)*sqrt(1-e**2) / (1-e*cos(E)))
        # alt: ta = arccos((cos(E)-e)/(1-e*cos(E)))
    elif e == 1 and B:
        ta = np.arsin(p*B/r)
        # alt: ta = (p-r)/r
    else: # e > 1 and H:
        ta = arcsin( (-sinh(H)*sqrt(e**2-1)) / (1-e*cosh(H)) )
        # alt: ta = arccos((cosh(ta)-e)/(1-e*cosh(H)))
    return ta


def univ_anomalies(M=None, e=None, dt=None, p=None, center='earth'):
    """universal formulation of elliptical, parabolic, and hyperbolic
    solutions for Kepler's Equation using Newton-Raphson's interation method
    where applicable.
    :param M: mean anomaly (rad)
    :param e: eccentricity
    :param dt: time of flight from orbit periapsis (s)
    :param p: semi-parameter (km)
    :return E: eccentric anomaly for elliptical orbits (rad)
    :return B: parabolic anomaly for parabolic orbits (rad)
    :return H: hyperbolic anomaly for hyperbolic orbits (rad)
    """
    mu = get_mu(center=center)

    # elliptical solution
    if e < 1 and M:
        if -np.pi < M < 0 or np.pi < M < 2*np.pi:
            E = M - e
        else:
            E = M + e
        count = 0
        E_prev = 0
        while (np.abs(E - E_prev) > 1e-15):
            if count == 0:
                count += 1
            else:
                E_prev = E
            E = E_prev + (M - E_prev + e*sin(E_prev)) / (1-e*cos(E_prev))
        return E

    # parabolic solution
    elif e == 1 and p:
        n_p = 2*sqrt(mu/p**3)
        s = 0.5*arctan(2/(3*n_p*dt))
        w = arctan(tan(s)**(1/3))
        B = 2/tan(2*w)
        return B

    # hyperbolic solution
    else: # e > 1
        if e < 1.6 and M:
            if -np.pi < M < 0 or np.pi < M < 2*np.pi:
                H = M - e
            else:
                H = M + e
        elif 1.6 <= e < 3.6 and np.abs(M) > np.pi:
            H = M - np.sign(M)*e
        else:
            H = M / (e-1)

        count = 0
        H_prev = 0
        while (np.abs(H - H_prev) > 1e-15):
            if count == 0:
                count += 1
            else:
                H_prev = H
            H = H_prev + (M - e*sinh(H_prev)+H_prev) / (e*cosh(H_prev)-1)
        return H


def semimajor_axis(p, e):
    """returns semi-major axis for a Keplerian orbit
    :param p:
    :param e:
    :return a:
    """
    a = p / (1-e**2)

    return a


def traj_equation(p, e, tanom):
    """returns trajectory equation for a Keplerian orbit
    :param p:
    :param e:
    :param tanom:
    :return r:
    """
    r = p / ( 1 + e*cos(tanom))

    return r


def vel_mag(r=None, a=None, e=None, p=None, tanom=None, center='earth'):
    """returns orbital velocity at true anomaly point
    :param r:
    :param a:
    :param e:
    :param p:
    :param tanom:
    :param center:
    :return vmag:
    not fully tested
    """
    mu = get_mu(center=center)

    if r and a:
        vmag = sqrt(2*mu/r - mu/a)
        return vmag
    elif e and p and tanom:
        a = semimajor_axis(p, e)
        r = traj_equation(p, e, tanom)
        vmag = sqrt(2*mu/r - mu/a)
        return vmag
    else:
        return "Need at least r, a; or e, p, tanom"


class Keplerian(object):
    """Class to compute classical Keplerian elements from
    position/velocity vectors. 
    :param rvec: positional vectors of spacecraft (km)
    :param vvec: velocity vectors of spacecraft (km/s)
    :param center: center object of orbit; default=earth
    """

    def __init__(self, rvec, vvec, center='earth'):

        # determine gravitational constant
        self.mu = get_mu(center=center)

        # position and veloccity
        self.rvec = rvec
        self.vvec = vvec
        self.r_mag = vec.norm(rvec)
        self.v_mag = vec.norm(vvec)

        # angular momentun; orbit normal direction
        self.h_vec = vec.vcrossv(rvec, vvec)
        self.h_mag = vec.norm(self.h_vec)

        # node vector K; n = 0 for equatorial orbits
        self.node_vec = vec.vcrossv(v1=[0,0,1], v2=self.h_vec)
        self.node_mag = vec.norm(self.node_vec)

        # eccentricity vector; e = 0 for circular orbits
        self.e_vec = self.eccentricity_vector
        self.e_mag = vec.norm(self.e_vec)


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
        return arccos(self.h_vec[2]/self.h_mag)


    @property
    def true_anomaly(self):
        """true anomaly"""
        if self.e_mag == 0:
            return float('nan')
        ta = arccos(vec.vdotv(self.e_vec, self.rvec)/(self.e_mag*self.r_mag))
        if vec.vdotv(self.rvec, self.vvec) < 0:
            ta = 2*np.pi - ta
        return ta


    @property
    def raan(self):
        """right ascending node"""
        if self.inclination == 0:
            return float('nan')
        omega = arccos(self.node_vec[0]/self.node_mag)
        if self.node_vec[1] < 0:
            omega = 2*np.pi - omega
        return omega


    @property
    def aop(self):
        """argument_of_periapse"""
        if self.e_mag == 0 or self.node_mag == 0:
            return float('nan')
        argp = arccos(vec.vdotv(self.node_vec, self.e_vec)/(self.node_mag*self.e_mag))
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

    # get r,v from orbital elements
    rv = get_rv_frm_elements(p=14351, e=0.5, i=np.rad2deg(45), raan=np.rad2deg(30), aop=0, ta=0)
    print(rv) # r=[9567.2, 0], v=[0, 7.9054]

    # testing specifc energy function
    pos = vec.vxs(scalar=1e4, v1=[1.2756, 1.9135, 3.1891]) # km
    vel = [7.9053, 15.8106, 0.0] # km/s
    energy = sp_energy(vel=vel, pos=pos, mu=get_mu(center='earth'))
    print(energy)


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


    # example from pg 114 vallado
    # orbital positon/velocity
    r = [6524.834, 6862.875, 6448.296]
    v =  [4.901327, 5.533756, -1.976341]
    elements = get_orbital_elements(rvec=r, vvec=v)
    print(elements)

    # test get_rv_frm_elements(p, e, i, raan, aop, ta, center='earth'):
    p = 11067.79
    e = 0.83285
    i = np.deg2rad(87.87)
    raan = np.deg2rad(227.89)
    aop = np.deg2rad(53.38)
    ta = np.deg2rad(92.335)
    r, v = get_rv_frm_elements(p, e, i, raan, aop, ta, center='earth')
    print(r) # [6525.36812099 6861.5318349  6449.11861416]
    print(v) # [ 4.90227865  5.53313957 -1.9757101 ]

    E = univ_anomalies(e=0.4, M=np.deg2rad(235.4))
    print(E) # 3.84866174509717 rad

    B = univ_anomalies(e=1, dt=53.7874*60, p=25512)
    print(B) # 0.817751

    H = univ_anomalies(e=2.4, M=np.deg2rad(235.4))
    print(H) # 1.6013761449