#!/usr/bin/env python3

import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import vectors as vec
from math_helpers import rotations as rot
from math_helpers import matrices as mat

# gravitational constants
mu = 398600.4418 # earth

# earth constants
REq_earth = 6378.1363 # (km)
RPolar_earth = 6356.7516005 # (km) 
F_earth = 0.003352813178 # = 1/298.257
E_earth = 0.081819221456
omega_earth = 7.292115e-5 # +/- 1.5e-12 (rad/s)


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
        r1, r2 = [r+REq_earth for r in [r1, r2]]

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


if __name__ == '__main__':
    
    #
    r1 = 7028.137
    r2 = 42158.137
    print(hohmann_transfer(r1, r2))