#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import rotations, vectors, quaternions, matrices
from traj import conics
from traj import maneuvers as man
import numpy as np


def test_coplanar_transfer():
    """tests general coplanar orbit transfer
    """
    r1 = 12750
    r2 = 31890
    p = 13457
    e = 0.76
    dv1, dv2 = man.coplanar_transfer(p, e, r1, r2, center='earth')
    dv_tot_truth = 7.086
    dv_tot = dv1 + dv2
    assert np.allclose(dv_tot, dv_tot_truth,  atol=1e-03)


def test_hohmann_transfer():
    """
    """
    alt1 = 191.34411
    alt2 = 35781.34857
    dv1, dv2, tt = man.hohmann_transfer(alt1, alt2, use_alts=True, center='earth')
    tt = tt/60.
    dv_tot = np.abs(dv1) + np.abs(dv2)
    dv_tot_truth = 3.935224 # km/s
    dt_truth = 315.403 # min 
    assert np.allclose(dv_tot, dv_tot_truth)
    assert np.allclose(tt, dt_truth)

    # given radii of two orbits
    r1 = 400
    r2 = 800
    # compute and test delta v required for a hohmann transfer
    dv1, dv2, trans_time = man.hohmann_transfer(r1, r2, use_alts=True, center='earth')
    dv1_truth = 0.1091 # km/s
    dv2_truth = 0.1076 # km/s
    assert np.allclose(dv1, dv1_truth,rtol=0, atol=1e-04)
    assert np.allclose(dv2, dv2_truth,rtol=0, atol=1e-04)


def test_bielliptic():
    """
    """
    alt1 = 191.34411
    altb = 503873
    alt2 = 376310
    dv1, dv_trans, dv2, tt = \
            man.bielliptic_transfer(alt1, alt2, altb, use_alts=True, center='Earth')
    dv1_truth = 3.156233389
    dv2_truth = -0.070465937
    dv_trans_truth = 0.677357998
    tt_truth = 593.92000 # hrs
    assert np.allclose(dv1, dv1_truth)
    assert np.allclose(dv_trans, dv_trans_truth)
    assert np.allclose(dv2, dv2_truth)
    assert np.allclose(tt/3600, tt_truth)
