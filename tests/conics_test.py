#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import matrices as mat
from math_helpers import rotations, vectors, quaternions, matrices
from orbitals import conics
import numpy as np

def test_hohmann_transfer():
    """tests matrix transpose and multiplication
    """
    # given radii of two orbits
    r1 = 400
    r2 = 800
    # compute and test delta v required for a hohmann transfer
    dv1, dv2 = conics.hohmann_transfer(r1, r2, object='earth')
    dv1_truth = 0.1091 # km/s
    dv2_truth = 0.1076 # km/s
    assert np.allclose(dv1, dv1_truth,rtol=0, atol=1e-04)
    assert np.allclose(dv2, dv2_truth,rtol=0, atol=1e-04)

def test_coplanar_transfer():
    """tests general coplanar orbit transfer
    """
    r1 = 12750
    r2 = 31890
    p = 13457
    e = 0.76
    dv1, dv2 = conics.coplanar_transfer(p, e, r1, r2)
    dv_tot_truth = 7.086
    dv_tot = dv1 + dv2
    assert np.allclose(dv_tot, dv_tot_truth,  atol=1e-03)