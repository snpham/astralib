#! python3
import spiceypy as spice
import numpy as np
from numpy.linalg import norm


if __name__ == "__main__":

    #1.
    spice.furnsh('lessons/insitu_sensing/kernels/lsk/naif0012.tls')
    cass = -82
    utc = '2004-06-11T19:32:00'
    et = spice.utc2et(utcstr=utc)
    print(f'Date and Time: {utc}')
    print(f'Insitu Sensing: et, = {et}')

    #2.
    spice.furnsh('lessons/insitu_sensing/kernels/sclk/cas00084.tsc')
    scepch = '1465674964.105'
    scet = spice.scs2e(sc=-82, sclkch=scepch)
    print(f'CASSINI clock epoch = {scet}')

    #3.
    spice.furnsh([#'lessons/insitu_sensing/kernels/sclk/cas00084.tsc',
                  #'lessons/insitu_sensing/kernels/pck/cpck05Mar2004.tpc',
                  'lessons/insitu_sensing/kernels/spk/030201AP_SK_SM546_T45.bsp',
                  'lessons\insitu_sensing\kernels\spk\981005_PLTEPH-DE405S.bsp'])
    cs_state = spice.spkezr(targ=str(cass), et=scet, ref='eclipj2000', abcorr='none', obs='sun')[0]
    print(f'position relative to Sun: [x,y,z]=[{cs_state[0]:0.6f}, {cs_state[1]:0.6f}, {cs_state[2]:0.6f}]km')
    print(f'velocity rela to Sun:  [vx,vy,vz]=[{cs_state[3]:0.6f}, {cs_state[4]:0.6f}, {cs_state[5]:0.6f}]km/s')
    spice.kclear()

    #4.
    spice.furnsh(['lessons/insitu_sensing/kernels/lsk/naif0012.tls',
                  'lessons/insitu_sensing/kernels/sclk/cas00084.tsc',
                  #'lessons/insitu_sensing/kernels/pck/cpck05Mar2004.tpc',
                  'lessons\insitu_sensing\kernels\ck\\04135_04171pc_psiv2.bc',
                  'lessons\insitu_sensing\kernels\\fk\cas_v37.tf',
                  'lessons/insitu_sensing/kernels/spk/030201AP_SK_SM546_T45.bsp',
                  'lessons\insitu_sensing\kernels\spk\981005_PLTEPH-DE405S.bsp'])
    cs2sun_pos = spice.spkpos(targ='sun', et=scet, ref='CASSINI_INMS', abcorr='LT+S', obs='CASSINI')[0]
    print(f'sunpos relative to cass:  [x,y,z]=[{cs2sun_pos[0]:0.6f}, {cs2sun_pos[1]:0.6f}, {cs2sun_pos[2]:0.6f}]km')
    cs2sun_pos_hat = spice.vhat(cs2sun_pos)
    print(f'unit vectors: [x,y,z]=[{cs2sun_pos_hat[0]:0.6f}, {cs2sun_pos_hat[1]:0.6f}, {cs2sun_pos_hat[2]:0.6f}]')
    spice.kclear()

    #5.
    spice.furnsh(['lessons/insitu_sensing/kernels/lsk/naif0012.tls',
                  'lessons/insitu_sensing/kernels/sclk/cas00084.tsc',
                  'lessons/insitu_sensing/kernels/pck/cpck05Mar2004.tpc',
                  'lessons\insitu_sensing\kernels\ck\\04135_04171pc_psiv2.bc',
                  'lessons\insitu_sensing\kernels\\fk\cas_v37.tf',
                  'lessons\insitu_sensing\kernels\spk\\020514_SE_SAT105.bsp',
                  'lessons/insitu_sensing/kernels/spk/030201AP_SK_SM546_T45.bsp',
                  'lessons\insitu_sensing\kernels\spk\981005_PLTEPH-DE405S.bsp'])
    spoint, _, surfvec = spice.subpnt(method='NEAR POINT: ELLIPSOID', target='phoebe', et=scet, fixref='IAU_PHOEBE', abcorr='none', obsrvr='cassini')
    # print(f'Sub-spacecraft point on Phoebe (IAU_Phoebe) [xyz] = {spoint} km')
    # print(f'Vector from Cassini to sub-spacecraft point [xyz] = {surfvec} km')
    rho, lon, lat = spice.reclat(rectan=spoint)
    lon = np.rad2deg(lon)
    lat = np.rad2deg(lat)
    print(f'Sub-spacecraft point on Phoebe [lon, lat, rho] = [{lon:0.6f} deg, {lat:0.6f} deg, {rho:0.6f}] km')
    surfvec_hat = surfvec / np.linalg.norm(surfvec)
    T_ph2inms = spice.pxform(fromstr='IAU_PHOEBE', tostr='CASSINI_INMS', et=scet)
    surfvec_hat_inms = spice.mxv(m1=T_ph2inms, vin=surfvec_hat)
    print(f'Vector from Cassini to sub-spacecraft point (INMSf) [xyz] = {surfvec_hat_inms}')

    #6
    state_cass = spice.spkezr(targ='cassini', et=scet, ref='j2000', abcorr='none', obs='phoebe')[0][3:6]
    T_j2000toinms = spice.pxform(fromstr='j2000', tostr='cassini_inms', et=scet)
    state_cass = spice.mxv(m1=T_j2000toinms, vin=state_cass)
    vel_cass = state_cass
    vel_cass_hat = vel_cass / np.linalg.norm(vel_cass)
    print(f'Velocity of Cassini relative to Phoebe = {vel_cass_hat}')