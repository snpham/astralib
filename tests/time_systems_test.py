#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import rotations, vectors, quaternions, matrices, time_systems
from traj import conics
import numpy as np


def test_get_JD():
    """
    """
    jd, mjd = time_systems.get_JD(1996, 10, 26, 14, 20, 0)
    jd_truth = 2450383.09722222
    assert np.allclose(jd, jd_truth)

    