import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers.constants import *
import spiceypy as spice
from traj import maneuvers
from traj.meeus_alg import meeus
from traj.conics import get_rv_frm_elements2
import pandas as pd
from math_helpers.time_systems import get_JD


def launchwindows(departure_planet, departure_date, arrival_planet, 
                  arrival_window, dm=None, center='sun', run_test=False):
    """return plots of c3 and vinf values for a 0 rev lambert transfer 
    between a departure and arrival planet within a given arrival window.
    :param departure_planet: name of departure planet (str)
    :param departure_date: date of departure ('yyyy-mm-dd')
    :param arrival_planet: name of arrival planet (str)
    :param arrival_window: list with begin and end arrival date window 
                           ['yyyy-mm-dd', 'yyyy-mm-dd']
    :param dm: direction of motion (optional); if None, then the script 
               will auto-compute direction based on the change in true 
               anomaly
    :param center: point where both planets are orbiting about; 
                   default = 'sun'
    :param run_test: run unit tests with lower days and bypass plots
    """
    
    spice.furnsh(['spice/kernels/solarsystem/naif0012.tls'])

    # reinitializing
    dep_date = departure_date
    dp = departure_planet
    ap = arrival_planet

    # get departure julian dates
    dep_date = pd.to_datetime(dep_date)
    dep_JD = get_JD(dep_date.year, dep_date.month, dep_date.day, \
                          dep_date.hour, dep_date.minute, dep_date.second)

    days = 1000
    if run_test:
        days = 300

    # get arrival windows
    arrival_window = pd.to_datetime(arrival_window)
    arrival_window = np.linspace(arrival_window[0].value, arrival_window[1].value, days)
    arrival_window = pd.to_datetime(arrival_window)

    # get state of departure planet
    dep_elements = meeus(dep_JD, planet=dp)
    s_dep_planet = get_rv_frm_elements2(dep_elements, center=center)
    r_dep_planet = s_dep_planet[:3]
    v_dep_planet = s_dep_planet[3:6]

    # initializing arrival dataframe
    columns = ['tof_d', 'TOF', 'v_inf_dep', 'v_inf_arr', 'c3', 'vinf']
    transfer = pd.DataFrame(index=arrival_window, columns=columns)

    for date in arrival_window:

        transfer['tof_d'][date] = date.date()

        # get state of arrival planet at the current arrival date
        arrival_jdate = get_JD(date.year, date.month, date.day, \
                               date.hour, date.minute, date.second)
        et = spice.utc2et(str(date))
        jd = float(spice.et2utc(et, 'J', 10)[3:])
        assert np.allclose(arrival_jdate, jd)

        arr_elements = meeus(arrival_jdate, planet=ap)
        s_arrival = get_rv_frm_elements2(arr_elements, center=center)
        r_arr_planet = s_arrival[:3]
        v_arr_planet = s_arrival[3:6]

        # convert date since departure to seconds
        transfer['TOF'][date] = (date - dep_date).total_seconds()

        # compute lambert solution at current arrival date
        vi, vf = maneuvers.lambert_univ(r_dep_planet, r_arr_planet, \
                                        transfer['TOF'][date], dm=dm, 
                                        center=center, 
                                        dep_planet=dp, arr_planet=ap)

        # compute hyperbolic departure/arrival velocities
        transfer['v_inf_dep'][date] = vi - v_dep_planet
        transfer['v_inf_arr'][date] = vf - v_arr_planet
        
        # compute c3 values at departure and v_inf values at arrival
        transfer['c3'][date] = norm(transfer['v_inf_dep'][date])**2
        transfer['vinf'][date] = norm(transfer['v_inf_arr'][date])

    # get values and dates of min c3/v_inf
    minc3 = transfer['c3'].min()
    minc3_date = transfer['TOF'][transfer['c3'] == minc3]
    minvinf = transfer['vinf'].min()
    minvinf_date = transfer['TOF'][transfer['vinf'] == minvinf]
    print(f'(a) min c3 = {minc3} km2/s2 on {minc3_date.index[0]}'
          f' // {transfer.loc[transfer["c3"] == minc3, "tof_d"][0]}')
    print(f'(b) min v_inf = {minvinf} km/s on {minvinf_date.index[0]}'
          f' // {transfer.loc[transfer["vinf"] == minvinf, "tof_d"][0]}')

    if run_test:
        return minc3, minvinf, \
               str(minc3_date.index[0])[:10], str(minvinf_date.index[0])[:10]
                
    # # assuming positions of planets are in the ecliptic,
    # # determine Type 1 or 2 transfer
    tanom1 = np.arctan2(r_dep_planet[1], r_dep_planet[0])
    tanom2 = np.arctan2(r_arr_planet[1], r_arr_planet[0])
    dtanom = tanom2 - tanom1
    if dtanom > np.pi:
        ttype = '2'
    elif dtanom < np.pi:
        ttype = '1'

    # plots
    fig=plt.figure(figsize=(12,6))
    plt.style.use('seaborn')

    # c3 vs tof
    ax=fig.add_subplot(121)
    ax.set_xlabel(f"days past departure ({departure_date})")
    ax.set_ylabel("c3, km2/s2")
    ax.set_title(f"c3 versus time of flight, Type {ttype}")
    ax.plot(transfer['TOF']/3600/24, transfer['c3'], label='departure_c3')
    ax.plot(minc3_date.values/3600/24, minc3, 'bo', markersize=12, label='min_c3')
    # v_inf vs tof
    ax2=fig.add_subplot(122)
    ax2.set_xlabel(f"days past departure ({departure_date})")
    ax2.set_ylabel(f"v_inf at {ap}, km/s")
    ax2.set_title(f"v_inf at {ap} versus time of flight, Type {ttype}")
    ax2.plot(transfer['TOF']/3600/24, transfer['vinf'], label='arrival_vinf')
    ax2.plot(minvinf_date.values/3600/24, minvinf, 'ro', markersize=12, label='min_vinf')
    ax.legend()
    ax2.legend()
    fig.tight_layout(pad=4.0)
    plt.show()


if __name__ == '__main__':
    
    # # HW 2

    # problem 1
    departure_date = '2018-05-01 12:00'
    arrival_window = ['2018-09-01', '2018-12-20'] # 6-8 months form launch
    departure_planet = 'earth'
    arrival_planet = 'mars'
    launchwindows(departure_planet, departure_date,
                        arrival_planet, arrival_window,
                        dm=None, center='sun')

    # problem 1.2
    departure_date = '2018-05-01 12:00'
    arrival_window = ['2019-01-05', '2019-09-01'] # 6-8 months form launch
    departure_planet = 'earth'
    arrival_planet = 'mars'
    launchwindows(departure_planet, departure_date,
                        arrival_planet, arrival_window,
                        dm=None, center='sun')
                        

    # # Lecture 4b, slides 4
    # departure_date = '2005-06-04'
    # arrival_window = ['2005-07-04', '2005-12-04'] # 6-8 months form launch
    # departure_planet = 'earth'
    # arrival_planet = 'mars'
    # dm = None
    # launchwindows(departure_planet, departure_date,
    #                     arrival_planet, arrival_window,
    #                     dm=dm, center='sun')

    # # Lecture 4b, slides 5
    # departure_date = '2005-06-04'
    # arrival_window = ['2006-03-20', '2006-11-04'] # 6-8 months form launch
    # departure_planet = 'earth'
    # arrival_planet = 'mars'
    # dm = -1
    # launchwindows(departure_planet, departure_date,
    #                     arrival_planet, arrival_window,
    #                     dm=dm, center='sun')

    # # Lecture 4b, slides 6
    # departure_date = '2005-06-04'
    # arrival_window = ['2005-09-14', '2006-11-04'] # 6-8 months form launch
    # departure_planet = 'earth'
    # arrival_planet = 'mars'
    # dm = None
    # launchwindows(departure_planet, departure_date,
    #                     arrival_planet, arrival_window,
    #                     dm=dm, center='sun')

    # # # Lecture 4b, slides 9
    # departure_date = '2018-05-01'
    # arrival_window = ['2018-07-01', '2019-11-20'] # 6-8 months form launch
    # departure_planet = 'earth'
    # arrival_planet = 'mars'
    # dm = None
    # launchwindows(departure_planet, departure_date,
    #                     arrival_planet, arrival_window,
    #                     dm=dm, center='sun')


    # Hw2, p2
    # prob2_dawn()