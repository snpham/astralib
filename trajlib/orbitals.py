#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import matrices, quaternions, vectors
from math_helpers import mass_properties as mp
import numpy as np


def sp_energy(vel, pos, mu=398600.4418):
    """returns specific mechanical energy (km2/s2), angular momentum
    (km2/s), and flight-path angle (deg) of an orbit; 2body problem
    """
    v_mag =  np.linalg.norm(vel)
    r_mag = np.linalg.norm(pos)
    sp_energy =v_mag**2/2. - mu/r_mag
    ang_mo = vectors.vcrossv(v1=pos, v2=vel)
    if np.dot(a=pos, b=vel) > 0:
        phi = np.rad2deg(np.arccos(np.linalg.norm(ang_mo)/(r_mag*v_mag)))
    return sp_energy, ang_mo, phi



if __name__ == "__main__":
    # testing specifc energy function
    pos = vectors.vxs(scalar=1e4, v1=[1.2756, 1.9135, 3.1891]) # km
    vel = [7.9053, 15.8106, 0.0] # km/s
    sp_energy = sp_energy(vel=vel, pos=pos, mu=398600.4418)
    print(sp_energy)