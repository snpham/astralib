#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import (rotations, vectors, quaternions, matrices, coordinates)
from traj import conics
import numpy as np


def test_lat2rec():
    """tests lat2rec function
    """
    lon = np.deg2rad(345. + 35/60. + 51/3600.)
    lat = np.deg2rad(-1 * (7. + 54/60. + 23.886/3600.))
    elev = 56/1000.
    # lat = np.deg2rad(-7.9066357)
    # lon = np.deg2rad(345.5975)
    r = coordinates.lat2rec(lon, lat, elev, latref='geodetic', center='earth', ref='ellipsoid')
    r_truth = [6119.40026932, -1571.47955545, -871.56118090]
    assert np.allclose(r, r_truth, atol=1e-01)