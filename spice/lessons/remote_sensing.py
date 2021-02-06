#! python3
import spiceypy as spice
import numpy as np
from numpy.linalg import norm
from spiceypy.utils.support_types import SpiceyError


if __name__ == "__main__":

    #1
    # utctime = input('Input UTC Time: ')
    utctime = "2004 jun 11 19:32:00"
    spice.furnsh(['lessons\\remote_sensing\kernels\lsk\\naif0008.tls',
                  'lessons\\remote_sensing\kernels\sclk\cas00084.tsc'])
    sclkid = -82
    print(f'Converting UTC Time: {utctime}')
    et = spice.str2et(utctime)
    calet = spice.etcal(et=et)
    print(f'    Calendar ET: {calet}')
    calet = spice.timout(et=et, pictur='YYYY-MON-DDTHR:MN:SC ::TDB')
    print(f'    Calendar ET: {calet}')
    sclkst = spice.sce2s(sc=sclkid, et=et)
    print(f'    Spacecraft Clock Time: {sclkst}')
    julet = spice.et2utc(et=et, format_str='j', prec=2)
    print(f'    Julian Date: {julet}')
    dayet = spice.et2utc(et=et, format_str='d', prec=2)
    print(f'    Day-of-Year: {dayet}')

    #2
    spice.furnsh(['lessons\\remote_sensing\kernels\lsk\\naif0008.tls',
                  'lessons\\remote_sensing\kernels\spk/981005_PLTEPH-DE405S.bsp',
                  'lessons\\remote_sensing\kernels\spk\\020514_SE_SAT105.bsp',
                  'lessons\\remote_sensing\kernels\spk\\030201AP_SK_SM546_T45.bsp'])
    s_ph_cass = spice.spkezr(targ='phoebe', et=et, ref='j2000', abcorr='LT+S', obs='cassini')[0]
    print(f'Apparent state of Phoebe as seem from CASSINI in J2000\n'
          f'    [x,y,z] = {s_ph_cass[0:3]} km\n'
          f'    [vx,vy,vz] = {s_ph_cass[3:6]} km/s')
    s_ea_cass, lt_e_cass = spice.spkpos(targ='earth', et=et, ref='j2000', abcorr='lt+s', obs='cassini')
    print(f'Apparent state of Earth as seem from CASSINI in J2000\n'
          f'    [x,y,z]    = {s_ea_cass} km\n'
          f'    Light-time = {lt_e_cass:0.4f} s')
    s_sun_ph = spice.spkpos(targ='sun', et=et, ref='j2000', abcorr='LT+S', obs='phoebe')[0]
    print(f'Apparent state of Sun as seem from Phoebe in J2000\n'
          f'    [x,y,z]    = {s_sun_ph} km')
    dist = norm(s_sun_ph)
    dist_au = spice.convrt(dist, 'km', 'au')
    print(f'Actual distance b/t Sun and Phoebe: {dist:0.6f}km, {dist_au:0.6f} AU')

    #3
    spice.furnsh(['lessons/remote_sensing/kernels/lsk/naif0008.tls',
                  'lessons/remote_sensing/kernels/sclk/cas00084.tsc',
                  'lessons/remote_sensing/kernels/spk/981005_PLTEPH-DE405S.bsp',
                  'lessons/remote_sensing/kernels/spk/020514_SE_SAT105.bsp',
                  'lessons/remote_sensing/kernels/spk/030201AP_SK_SM546_T45.bsp',
                  'lessons/remote_sensing/kernels/fk/cas_v37.tf',
                  'lessons/remote_sensing/kernels/ck/04135_04171pc_psiv2.bc',
                  'lessons/remote_sensing/kernels/pck/cpck05Mar2004.tpc' ])
    s_ph_cass, ltime = spice.spkezr(targ='phoebe', et=et, ref='j2000', abcorr='LT+S', obs='cassini')
    T_ph2i = spice.sxform(instring='j2000', tostring='iau_phoebe', et=et)
    s_ph_cass_phf = spice.mxvg(m1=T_ph2i, v2=s_ph_cass, nrow1=6, nc1r2=6)
    print(f'Apparent state of Phoebe as seen from CASSINI in IAU_Phoebe\n'
          f'    [x,y,z]    = {s_ph_cass_phf[0:3]} km\n'
          f'    [vx,vy,vz] = {s_ph_cass_phf[3:6]} km/s')
    s_ph_cass, ltime = spice.spkezr(targ='phoebe', et=et, ref='iau_phoebe', abcorr='LT+S', obs='cassini')
    print(f'    [x,y,z]    = {s_ph_cass_phf[0:3]} km\n'
          f'    [vx,vy,vz] = {s_ph_cass_phf[3:6]} km/s')
    
    r_earth_cass = spice.spkpos(targ='earth', et=et, ref='j2000', abcorr='lt+s', obs='cassini')[0]
    bsight = [0.0, 0.0, 1.0]
    T_hga2i = spice.pxform(fromstr='cassini_hga', tostr='j2000', et=et)
    bsight = spice.mxv(m1=T_hga2i, vin=bsight)
    sep = np.rad2deg(spice.vsep(v1=bsight, v2=r_earth_cass))
    print(f'Angular separation b/t apparent positions of Earth and Cassini HGA:\n'
          f'    {sep:0.4f} deg')
    r_earth_cass = spice.spkpos(targ='earth', et=et, ref='cassini_hga', abcorr='lt+s', obs='cassini')[0]
    bsight = [0.0, 0.0, 1.0]
    sep = np.rad2deg(spice.vsep(v1=bsight, v2=r_earth_cass))
    print(f'    {sep:0.4f} deg')

    #4
    spice.furnsh(['lessons/remote_sensing/kernels/lsk/naif0008.tls',
                  'lessons/remote_sensing/kernels/spk/981005_PLTEPH-DE405S.bsp',
                  'lessons/remote_sensing/kernels/spk/020514_SE_SAT105.bsp',
                  'lessons/remote_sensing/kernels/spk/030201AP_SK_SM546_T45.bsp',
                  'lessons/remote_sensing/kernels/pck/cpck05Mar2004.tpc',
                  'lessons/remote_sensing/kernels/dsk/phoebe_64q.bds' ])
    for i in range(2):
        if i == 0:
            method = 'NEAR POINT/Ellipsoid'
        else:
            method = 'NADIR/DSK/Unprioritized'
        print(f'Subpoint/target shape model: {method}')

        spoint, trgepc, srfvec = spice.subpnt(method=method, target='phoebe', et=et, fixref='iau_phoebe', 
                                             abcorr='lt+s', obsrvr='cassini')
        print(f'    Apparent sub-observer point of Cassini on Phoebe, IAU_Phoebe\n'
              f'    [x,z,y] = {spoint} km, alt = {norm(srfvec)} km')

    #5
    spice.furnsh(['lessons/remote_sensing/kernels/lsk/naif0008.tls',
                  'lessons/remote_sensing/kernels/sclk/cas00084.tsc',
                  'lessons/remote_sensing/kernels/spk/981005_PLTEPH-DE405S.bsp',
                  'lessons/remote_sensing/kernels/spk/020514_SE_SAT105.bsp',
                  'lessons/remote_sensing/kernels/spk/030201AP_SK_SM546_T45.bsp',
                  'lessons/remote_sensing/kernels/fk/cas_v37.tf',
                  'lessons/remote_sensing/kernels/ck/04135_04171pc_psiv2.bc',
                  'lessons/remote_sensing/kernels/pck/cpck05Mar2004.tpc',
                  'lessons/remote_sensing/kernels/ik/cas_iss_v09.ti',
                  'lessons/remote_sensing/kernels/dsk/phoebe_64q.bds' ])
    room = 4
    try:
        nacid = spice.bodn2c(name='cassini_iss_nac')
    except SpiceyError:
        print('unable to locate the ID code for "Cassini_ISS_NAC"')
        raise
    shape, frame, bsight, n, fovbounds = spice.getfov(instid=nacid, room=room)
    fovbounds = fovbounds.tolist()
    fovbounds.append(bsight)
    vecnam = ['Bound 1', 'Bound 2', 'Bound 3', 'Bound 4', 'Cassini NAC Bsight']
    methods = ['Ellipsoid', 'DSK/Unprioritized']
    try:
        phoeid = spice.bodn2c(name='phoebe')
    except:
        print('Unable to locate the body ID code for Phoebe')
        raise
    for i in range(len(fovbounds)):
        print(f'Vector: {vecnam[i]}')
        for j in range(len(methods)):
            print(f'Target shape model: {methods[j]}')
            try:
                point, trgepc, srfvec = spice.sincpt(method=methods[j], target='phoebe', et=et, 
                                                     fixref='iau_phoebe', abcorr='lt+s', 
                                                     obsrvr='cassini', dref=frame, dvec=fovbounds[i])
                print(f'    Position vector of surface intercept in IAU_Phoebe\n'
                      f'    [x,y,z] = {point} km')
                rho, lon, lat = spice.reclat(rectan=point)
                print(f'    lat = {np.rad2deg(lon)} deg, lon = {np.rad2deg(lat)} deg')
                targepc, srfvec, phase, solar, emissn, visibl, lit = \
                                            spice.illumf(method=methods[j], target='phoebe', ilusrc='sun',
                                            et=et, fixref='iau_phoebe', abcorr='lt+s', obsrvr= 'cassini', spoint=point)
                print(f'    Phase angle = {np.rad2deg(phase)}\n'
                      f'    Solar incidence angle = {np.rad2deg(solar)}\n'
                      f'    Emission angle = {np.rad2deg(emissn)}\n'
                      f'    Observer visible: {visibl}\n'
                      f'    Sun visible: {lit}')
                if i == 4:
                    hr, mn, sc, time, ampm = spice.et2lst(et=trgepc, body=phoeid, lon=lon, typein='planetocentric')
                    print(f'    Local solar time at bsight intercept (24h): {time}')
                    print(f'    {ampm}')
            except SpiceyError as exc:
                print(f'    Exception message: {exc.value}')
            