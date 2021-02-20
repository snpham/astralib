#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import rotations, vectors, quaternions, matrices, time_systems
from traj import conics
import numpy as np
import pandas as pd


def test_get_JD():
    """tests get_jd function
    """
    jd = time_systems.get_JD(1996, 10, 26, 14, 20, 0)
    jd_truth = 2450383.09722222
    assert np.allclose(jd, jd_truth)

    jd = time_systems.get_JD(year=1957, month=10, day=4, hour=19, min=26, sec=24)
    jd_truth = 2436116.31
    assert np.allclose(jd, jd_truth)

    # some test dates from ASEN6008 - Hw4 problem 3
    launch_cal = '1989-10-08 00:00:00'
    cal = pd.to_datetime(launch_cal)
    jd = time_systems.get_JD(cal.year, cal.month, cal.day, cal.hour, cal.minute, cal.second)
    launch_jd = 2447807.5
    assert np.allclose(jd, launch_jd)
    venus_flyby_cal = '1990-02-10 00:00:00'
    cal = pd.to_datetime(venus_flyby_cal)
    jd = time_systems.get_JD(cal.year, cal.month, cal.day, cal.hour, cal.minute, cal.second)
    venus_flyby_jd = 2447932.5
    assert np.allclose(jd, venus_flyby_jd)
    earth_flyby_cal = '1990-12-10 00:00:00'
    cal = pd.to_datetime(earth_flyby_cal)
    jd = time_systems.get_JD(cal.year, cal.month, cal.day, cal.hour, cal.minute, cal.second)
    earth_flyby_jd = 2448235.5
    assert np.allclose(jd, earth_flyby_jd)
    earth_flyby2_cal = '1992-12-09 12:00:00'
    cal = pd.to_datetime(earth_flyby2_cal)
    jd = time_systems.get_JD(cal.year, cal.month, cal.day, cal.hour, cal.minute, cal.second)
    earth_flyby2_jd = 2448966.0
    assert np.allclose(jd, earth_flyby2_jd)
    jupiter_arrival_cal = '1996-03-21 12:00:00'
    jupiter_arrival_jd = 2450164.0
    cal = pd.to_datetime(jupiter_arrival_cal)
    jd = time_systems.get_JD(cal.year, cal.month, cal.day, cal.hour, cal.minute, cal.second)
    assert np.allclose(jd, jupiter_arrival_jd)


def test_cal_from_jd():
    """tests cal_from_jd function
    """
    jd = 2449877.3458762
    date = time_systems.cal_from_jd(jd) 
    date_t = (1995, 6, 8, 20, 18, 3.70369)
    assert np.allclose(date, date_t)
