#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers.constants import *
from traj.missiondesign import launchwindows
import pytest


@pytest.mark.skip(reason="takes too long")
def test_launchwindows():
    """tests launchwindows
    """
    # ASEN6008 - HW 2
    # problem 1
    departure_date = '2018-05-01 12:00'
    arrival_window = ['2018-09-01', '2018-12-20']
    departure_planet = 'earth'
    arrival_planet = 'mars'
    c3, vinf, c3date, vinfdate = launchwindows(departure_planet, departure_date, arrival_planet, 
                            arrival_window, dm=None, center='sun', run_test=True)
    assert np.allclose([c3, vinf], [8.2622238789, 2.997071667])
    assert c3date == '2018-11-06' and vinfdate == '2018-11-27'

    # problem 1.2
    departure_date = '2018-05-01 12:00'
    arrival_window = ['2019-01-05', '2019-09-01']
    departure_planet = 'earth'
    arrival_planet = 'mars'
    c3, vinf, c3date, vinfdate = launchwindows(departure_planet, departure_date, arrival_planet, 
                            arrival_window, dm=None, center='sun', run_test=True)
    assert np.allclose([c3, vinf], [9.008745818, 3.53387411], atol=1e-02)
    assert c3date == '2019-01-11' and vinfdate == '2019-01-13'

    # Lecture 4b, slides 4
    departure_date = '2005-06-04'
    arrival_window = ['2005-07-04', '2005-12-04']
    departure_planet = 'earth'
    arrival_planet = 'mars'
    dm = None
    c3, vinf, c3date, vinfdate = launchwindows(departure_planet, departure_date, arrival_planet, 
                            arrival_window, dm=None, center='sun', run_test=True)
    assert np.allclose([c3, vinf], [47.626555237, 5.4104737309,])
    assert c3date == '2005-12-04' and vinfdate == '2005-12-04'

    # Lecture 4b, slides 5
    departure_date = '2005-06-04'
    arrival_window = ['2006-03-20', '2006-11-04']
    departure_planet = 'earth'
    arrival_planet = 'mars'
    dm = -1
    c3, vinf, c3date, vinfdate = launchwindows(departure_planet, departure_date, arrival_planet, 
                            arrival_window, dm=None, center='sun', run_test=True)
    assert np.allclose([c3, vinf], [39.236466566, 2.5910593022])
    assert c3date == '2006-03-20' and vinfdate == '2006-03-20'

    # Lecture 4b, slides 6
    departure_date = '2005-06-04'
    arrival_window = ['2005-09-14', '2006-11-04']
    departure_planet = 'earth'
    arrival_planet = 'mars'
    dm = None
    c3, vinf, c3date, vinfdate = launchwindows(departure_planet, departure_date, arrival_planet, 
                            arrival_window, dm=None, center='sun', run_test=True)
    assert np.allclose([c3, vinf], [38.013413641, 2.540104621], atol=1e-03)
    assert c3date == '2006-02-25' and vinfdate == '2006-03-05'

    # # Lecture 4b, slides 9
    departure_date = '2018-05-01'
    arrival_window = ['2018-07-01', '2019-11-20']
    departure_planet = 'earth'
    arrival_planet = 'mars'
    dm = None
    c3, vinf, c3date, vinfdate = launchwindows(departure_planet, departure_date, arrival_planet, 
                            arrival_window, dm=None, center='sun', run_test=True)
    assert np.allclose([c3, vinf], [8.30347037, 3.001589343], atol=1e-03)
    assert c3date == '2018-11-05' and vinfdate == '2018-11-27'
