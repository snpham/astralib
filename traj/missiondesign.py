import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers.constants import *
import spiceypy as spice
from traj import lambert
from traj.meeus_alg import meeus
from traj.conics import get_rv_frm_elements2
from traj.bplane import bplane_vinf
import pandas as pd
from math_helpers.time_systems import get_JD, cal_from_jd


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

        arr_elements = meeus(arrival_jdate, planet=ap)
        s_arrival = get_rv_frm_elements2(arr_elements, center=center)
        r_arr_planet = s_arrival[:3]
        v_arr_planet = s_arrival[3:6]

        # convert date since departure to seconds
        transfer['TOF'][date] = (date - dep_date).total_seconds()

        # compute lambert solution at current arrival date
        vi, vf = lambert.lambert_univ(r_dep_planet, r_arr_planet, \
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


def get_porkchops(dep_jd_init, dep_jd_fin, arr_jd_init, arr_jd_fin, 
                  dp='earth', ap='jupiter', center='sun', 
                  contour_tof=None, contour_c3=None,
                  contour_vinf=None, contour_vinf_out=None,
                  plot_tar=False, tar_dep=None, tar_arr=None,
                  shade_c3=False, shade_tof=False, shade_vinf_arr=False,
                  shade_vinf_range=None, shade_tof_range=None):
    """generates a porkchop plot for a given launch and arrival window.
    :param dep_jd_init: initial departure date (JD)
    :param dep_jd_fin: final departure date (JD)
    :param arr_jd_init: initial arrival date (JD)
    :param arr_jd_fin: final arrival date (JD)
    :param dp: departure planet
    :param ap: arrival planet
    :param center: center body of orbit; default='sun'
    :param contour_tof: array of tof contours to plot
    :param contour_c3: array of launch c3 contours to plot (optional)
    :param contour_vinf: array of vinf inbound contours to plot
    :param contour_vinf_out: array of vinf outbound contours to plot (optional)
    :param plot_tar: plot target point (True); default=False
    :param tar_dep: target departure date (JD)
    :param tar_arr: target arrival date (JD)
    :param shade_c3: option to shade certain c3 contours (True)
    :param shade_tof: option to shade certain tof contours (True)
    :param shade_vinf_arr: option to shade certain arrival vinf contours (True)
    :param shade_vinf_range: array of arrival vinf range to shade
    :param shade_tof_range: array of time of flight range to shade
    :return df: if contour_c3 is present, [df_tof, df_c3, df_vinf_arr];
                if contour_vinf_out is present, 
                [df_tof, df_vinf_dep, df_vinf_arr]
    """
    plot_c3 = True
    plot_vinf_out = True
    if contour_c3 is None:
        plot_c3 = False
    if contour_vinf_out is None:
        plot_vinf_out = False

    # departure and arrival dates
    dep_date_initial_cal = cal_from_jd(dep_jd_init, rtn='string')
    arr_date_initial_cal = cal_from_jd(arr_jd_init, rtn='string')
    dep_date_initial_cal = pd.to_datetime(dep_date_initial_cal)
    arr_date_initial_cal = pd.to_datetime(arr_date_initial_cal)

    # time windows
    delta_dep = dep_jd_fin - dep_jd_init
    delta_arr = arr_jd_fin - arr_jd_init
    departure_window = np.linspace(dep_jd_init, dep_jd_fin, int(delta_dep))
    arrival_window = np.linspace(arr_jd_init, arr_jd_fin, int(delta_arr))

    # generate dataframes for c3, time of flight, and dep/arrival v_inf
    df_c3 = pd.DataFrame(index=arrival_window, columns=departure_window)
    df_tof = pd.DataFrame(index=arrival_window, columns=departure_window)
    df_vinf_arr = pd.DataFrame(index=arrival_window, columns=departure_window)
    df_vinf_dep = pd.DataFrame(index=arrival_window, columns=departure_window)

    # loop through launch dates
    for dep_JD in departure_window:

        for arr_JD in arrival_window:

            tof_s = (arr_JD-dep_JD)*3600*24
            s_planet1 = meeus(dep_JD, planet=dp, rtn='states', ref_rtn=center)
            s_planet2 = meeus(arr_JD, planet=ap, rtn='states', ref_rtn=center)
            vi, vf = lambert.lambert_univ(s_planet1[:3], s_planet2[:3], tof_s, 
                                  center=center, dep_planet=dp, arr_planet=ap)
            
            c3 = norm(vi-s_planet1[3:6])**2
            vinf_arr = norm(vf - s_planet2[3:6])
            vinf_dep = norm(vi - s_planet1[3:6])
            df_c3[dep_JD][arr_JD] = c3
            df_tof[dep_JD][arr_JD] = arr_JD-dep_JD
            df_vinf_arr[dep_JD][arr_JD] = vinf_arr
            df_vinf_dep[dep_JD][arr_JD] = vinf_dep

    # generate contour plots
    fig, ax = plt.subplots(figsize=(10,8))
    CS_tof = ax.contour(departure_window-departure_window[0], arrival_window-arrival_window[0], 
                        df_tof, linewidths=0.5, colors=('gray'), levels=contour_tof)
    CS_vinf_arr = ax.contour(departure_window-departure_window[0], arrival_window-arrival_window[0], 
                             df_vinf_arr, linewidths=0.5, colors=('g'), levels=contour_vinf)
    if plot_vinf_out:
        CS_vinf_dep = ax.contour(departure_window-departure_window[0], arrival_window-arrival_window[0], 
                                 df_vinf_dep, linewidths=0.5, colors=('b'), levels=contour_vinf_out)
    if plot_c3:
        CS_c3 = ax.contour(departure_window-departure_window[0], arrival_window-arrival_window[0], 
                           df_c3, linewidths=0.5, colors=('b'), levels=contour_c3)

    ax.set_title(f'pork chop plot from {dp} to {ap}')
    ax.set_xlabel(f'{dp} departure dates - days since {dep_date_initial_cal}') 
    ax.set_ylabel(f'{ap} arrival dates - days since {arr_date_initial_cal}') 

    ax.clabel(CS_tof, inline=0.2, fmt="%.0f", fontsize=10)
    ax.clabel(CS_vinf_arr, inline=0.2, fmt="%.1f", fontsize=10)
    h1,_ = CS_tof.legend_elements()
    h3,_ = CS_vinf_arr.legend_elements()

    if plot_c3:
        ax.clabel(CS_c3, inline=0.2, fmt="%.1f", fontsize=10)
        h2,_ = CS_c3.legend_elements()
        ax.legend([h1[0], h2[0], h3[0]], ['TOF, days', 'c3, km2/s2', 'v_inf_arrival, km/s'], 
                loc=2, facecolor='white', framealpha=1)
    elif plot_vinf_out:
        ax.clabel(CS_vinf_dep, inline=0.2, fmt="%.1f", fontsize=10)   
        h2,_ = CS_vinf_dep.legend_elements()
        ax.legend([h1[0], h2[0], h3[0]], ['TOF, days', 'vinf_departure, km2/s2', 'v_inf_arrival, km/s'], 
                loc=2, facecolor='white', framealpha=1)

    if plot_tar:
        plt.scatter(tar_dep-dep_jd_init, tar_arr-arr_jd_init, linewidths=18, color='orange')

    # shade region within these bounds
    if shade_vinf_arr:
        CS_vinf_arr = ax.contourf(departure_window-departure_window[0], 
                              arrival_window-arrival_window[0], df_vinf_arr, 
                              colors=('g'), levels=shade_vinf_range, alpha=0.3)
    if shade_tof:
        CS_tof = ax.contourf(departure_window-departure_window[0], 
                             arrival_window-arrival_window[0], df_tof, 
                             colors=('black'), levels=shade_tof_range, alpha=0.3)
    
    plt.savefig(f'porkschops_{dp}_{ap}.png')
    plt.show()

    if plot_c3:
        return [df_tof, df_c3, df_vinf_arr]
    elif plot_vinf_out:
        return [df_tof, df_vinf_dep, df_vinf_arr]
    else:
        return [df_tof, df_vinf_arr]


def run_pcp_search(dep_jd_init, dep_jd_fin, pl2_jd_init, pl2_jd_fin, pl3_jd_init, pl3_jd_fin, 
                   dpl='earth', pl2='jupiter', pl3='pluto', center='sun', 
                   c3_max=None, vinf_max=None, vinf_tol=None, rp_min=None, fine_search=False):
    """generates a porkchop plot for a given launch and arrival window.
    :param dep_jd_init: initial departure date of launch planet (planet 1) (JD)
    :param dep_jd_fin: final departure date of launch planet (planet 1) (JD)
    :param pl2_jd_init: initial arrival date of flyby planet (planet 2) (JD)
    :param pl2_jd_fin: final arrival date of flyby planet (planet 2) (JD)
    :param pl3_jd_init: initial arrival date of arrival planet (planet 3) (JD)
    :param pl3_jd_fin: final arrival date of arrival planet (planet 3) (JD)
    :param dpl: name of departure planet (planet 1)
    :param pl2: name of flyby planet (planet 2)
    :param pl3: name of arrival planet (planet 3)
    :param center: center body of orbit; default='sun'
    :param c3_max: maximum launch c3 constraint (km2/s2)
    :param vinf_max: maximum final arrival vinf at planet 3 (km/s)
    :param vinf_tol: maximum allowable delta-vinf inbound/outbound of flyby (km/s)
    :param rp_min: minimum radius of flyby (km)
    :param fine_search: option between coarse search of 3 days interval (False);
                        or fine search of 0.8 days interval (True)
    :return df: [dfpl1_c3, dfpl2_tof, dfpl2_vinf_in, dfpl2_vinf_out, ...
                 dfpl3_tof, dfpl3_vinf_in, dfpl3_rp]
    in work, need to add more robustness for constraining options
    """
    # departure and arrival dates
    dep_date_init_cal = pd.to_datetime(cal_from_jd(dep_jd_init, rtn='string'))
    pl2_jd_init_cal = pd.to_datetime(cal_from_jd(pl2_jd_init, rtn='string'))
    pl3_jd_init_cal = pd.to_datetime(cal_from_jd(pl3_jd_init, rtn='string'))

    # time windows
    delta_dep = dep_jd_fin - dep_jd_init
    delta_pl2 = pl2_jd_fin - pl2_jd_init
    delta_pl3 = pl3_jd_fin - pl3_jd_init
    searchint = 3
    if fine_search:
        searchint = 0.8
    dep_window = np.linspace(dep_jd_init, dep_jd_fin, int(delta_dep/searchint))
    # print(dep_window)
    pl2_window = np.linspace(pl2_jd_init, pl2_jd_fin, int(delta_pl2/searchint))
    pl3_window = np.linspace(pl3_jd_init, pl3_jd_fin, int(delta_pl3/searchint))


    # generate dataframes for c3, time of flight, and dep/arrival v_inf
    dfpl1_c3 = pd.DataFrame(index=pl2_window, columns=dep_window)
    dfpl2_tof = pd.DataFrame(index=pl2_window, columns=dep_window)
    dfpl2_vinf_in = pd.DataFrame(index=pl2_window, columns=dep_window)
    dfpl2_vinf_out = pd.DataFrame(index=pl2_window, columns=dep_window)
    dfpl3_tof = pd.DataFrame(index=pl3_window, columns=pl2_window)
    dfpl3_vinf_in = pd.DataFrame(index=pl3_window, columns=pl2_window)
    dfpl3_rp = pd.DataFrame(index=pl3_window, columns=pl2_window)

    count = 0
    # loop through launch dates
    for dep_JD in dep_window:

        for arr_JD in pl2_window:

            tof12_s = (arr_JD-dep_JD)*3600*24
            s_planet1 = meeus(dep_JD, planet=dpl, rtn='states', ref_rtn=center)
            s_planet2 = meeus(arr_JD, planet=pl2, rtn='states', ref_rtn=center)
            vi_seg1, vf_seg1 = lambert.lambert_univ(s_planet1[:3], s_planet2[:3], tof12_s, 
                                  center=center, dep_planet=dpl, arr_planet=pl2)
            
            c3 = norm(vi_seg1-s_planet1[3:6])**2

            if c3 < c3_max:
                # print('c3', c3)
                for arr2_JD in pl3_window:
                    tof23_s = (arr2_JD-arr_JD)*3600*24
                    s_planet3 = meeus(arr2_JD, planet=pl3, rtn='states', ref_rtn=center)
                    vi_seg2, vf_seg2 = lambert.lambert_univ(s_planet2[:3], s_planet3[:3], tof23_s, 
                                        center=center, dep_planet=pl2, arr_planet=pl3)

                    vinf_pl2_in = norm(vf_seg1 - s_planet2[3:6])
                    vinf_pl2_out = norm(vi_seg2 - s_planet2[3:6])

                    if abs(vinf_pl2_in-vinf_pl2_out) < vinf_tol:
                        
                        # print(abs(vinf_pl2_in-vinf_pl2_out))

                        rp = bplane_vinf(vf_seg1, vi_seg2, center=pl2, rtn_rp=True)

                        if rp > rp_min:
                            
                            # print('rp', rp)

                            vinf_pl3_in = norm(vf_seg2 - s_planet3[3:6])

                            if vinf_pl3_in < vinf_max:

                                # print('vinf_pl2_out', vinf_pl2_out)

                                dfpl1_c3[dep_JD][arr_JD] = c3
                                dfpl2_tof[dep_JD][arr_JD] = arr_JD-dep_JD
                                dfpl2_vinf_in[dep_JD][arr_JD] = vinf_pl2_in
                                dfpl2_vinf_out[dep_JD][arr_JD] = vinf_pl2_out
                                dfpl3_tof[arr_JD][arr2_JD] = arr2_JD-arr_JD
                                dfpl3_vinf_in[arr_JD][arr2_JD] = vinf_pl3_in
                                dfpl3_rp[arr_JD][arr2_JD] = rp

    return [dfpl1_c3, dfpl2_tof, dfpl2_vinf_in, dfpl2_vinf_out, dfpl3_tof, dfpl3_vinf_in, dfpl3_rp]





if __name__ == '__main__':
    

    pass

    # # hw6 p1
    # # pcp from launch to jupiter gravity assist
    # dep_jd_init = 2453714.5
    # dep_jd_fin = 2453794.5
    # arr_jd_init = 2454129.5
    # arr_jd_fin = 2454239.5
    # dp = 'earth'
    # ap = 'jupiter'
    # center = 'sun'
    # contour_tof = np.arange(100,600,50)
    # contour_c3 = np.arange(100,300,10)
    # contour_vinf = np.arange(1,30,0.5)
    # plot_tar = True
    # tar_dep = 2453755.29167
    # tar_arr = 2454159.73681

    # # get_porkchops(dep_jd_init, dep_jd_fin, arr_jd_init, arr_jd_fin, 
    # #               dp=dp, ap=ap, center=center, 
    # #               contour_tof=contour_tof, contour_c3=contour_c3,
    # #               contour_vinf=contour_vinf, contour_vinf_out=None,
    # #               plot_tar=plot_tar, 
    # #               tar_dep=tar_dep, tar_arr=tar_arr,
    # #               shade_c3=False, shade_tof=False, shade_vinf=False,
    # #               shade_vinf_range=None, shade_tof_range=None)

    # # pcp from  jupiter gravity assist to PCE
    # dep_jd_init = 2454129.5
    # dep_jd_fin = 2454239.5
    # arr_jd_init = 2456917.5 
    # arr_jd_fin = 2457517.5
    # dp = 'jupiter'
    # ap = 'pluto'
    # center = 'sun'
    # contour_tof = np.arange(2500,3500,100)
    # contour_vinf_arr = np.arange(1,20,0.3)
    # contour_vinf_dep = np.arange(10,30,0.3)
    # plot_tar = True
    # tar_dep = 2454159.73681
    # tar_arr = 2457217.99931

    # # get_porkchops(dep_jd_init, dep_jd_fin, arr_jd_init, arr_jd_fin, 
    # #               dp=dp, ap=ap, center=center, 
    # #               contour_tof=contour_tof, contour_c3=None,
    # #               contour_vinf=contour_vinf, contour_vinf_out=contour_vinf_dep,
    # #               plot_tar=plot_tar, 
    # #               tar_dep=tar_dep, tar_arr=tar_arr,
    # #               shade_c3=False, shade_tof=False, shade_vinf=False,
    # #               shade_vinf_range=None, shade_tof_range=None)

    # dep1_jd_init = 2453714.5
    # dep1_jd_fin = 2453794.5
    # arr1_jd_init = 2454129.5
    # arr1_jd_fin = 2454239.5
    # dpl = 'earth'
    # pl2 = 'jupiter'
    # center = 'sun'
    # dep2_jd_init = arr1_jd_init
    # dep2_jd_fin = arr1_jd_fin
    # arr3_jd_init = 2456917.5 
    # arr3_jd_fin = 2457517.5
    # pl3 = 'pluto'
    # c3_max = 180 # km2/s2
    # vinf_max = 14.5 # km/s
    # vinf_tol = 0.1
    # rp_min = 30*r_jupiter

    # dep1_jd_init = 2453730.5
    # dep1_jd_fin = 2453778.5
    # arr1_jd_init = 2454145
    # arr1_jd_fin = 2454179
    # dep2_jd_init = arr1_jd_init
    # dep2_jd_fin = arr1_jd_fin
    # arr3_jd_init = 2457074
    # arr3_jd_fin = 2457517.5

    # # run_pcp_search(dep1_jd_init, dep1_jd_fin, dep2_jd_init, dep2_jd_fin, arr3_jd_init, arr3_jd_fin, 
    # #                 dpl=dpl, pl2=pl2, pl3=pl3, center=center, c3_max=c3_max, vinf_max=vinf_max, vinf_tol=vinf_tol, rp_min=rp_min)

    # req1 = get_JD(2006, 1, 9, 12, 0, 0)
    # dep1_jd_init = req1-4
    # dep1_jd_fin = req1+4
    # arr1_jd_init = 2454145
    # arr1_jd_fin = 2454179
    # dep2_jd_init = arr1_jd_init
    # dep2_jd_fin = arr1_jd_fin
    # arr3_jd_init = 2457074
    # arr3_jd_fin = 2457517.5
    # vinf_tol = 0.05

    # c3_max = 180 # km2/s2
    # vinf_max = 14.5 # km/s
    # vinf_tol = 0.1
    # rp_min = 30*r_jupiter
    # fine_search = True

    # dfs = run_pcp_search(dep1_jd_init, dep1_jd_fin, dep2_jd_init,
    #                      dep2_jd_fin, arr3_jd_init, arr3_jd_fin, 
    #                      dpl=dpl, pl2=pl2, pl3=pl3, center=center, 
    #                      c3_max=c3_max, vinf_max=vinf_max, vinf_tol=vinf_tol, 
    #                      rp_min=rp_min, fine_search=fine_search)

