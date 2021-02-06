#! python3
import spiceypy as spice
import numpy as np
from numpy.linalg import norm
if __name__ == "__main__":


    #1.
    spice.furnsh('lessons/binary_pck/kernels/lsk/naif0012.tls')
    et = spice.utc2et(utcstr='2020 APR 05 08:00:00 UTC')
    print(f'\nMoon Rotation: et, s={et}')

    #2.
    kernels = ['lessons/binary_pck/kernels/spk/de432s.bsp',
                'lessons/binary_pck/kernels/pck/pck00010.tpc'
                ]
    spice.furnsh(kernels)
    earth_lclf_iau = spice.spkpos(targ='earth', et=et, ref='iau_moon', abcorr='CN+S', obs='moon')[0]
    radius, lon, lat = spice.reclat(rectan=earth_lclf_iau)
    lon = np.rad2deg(lon)
    lat = np.rad2deg(lat)
    print(f'iau_moon: radius={radius:.6f}km, lon={lon:.6f} deg, lat={lat:.6f} deg')
    spice.kclear()

    #3.
    kernels = ['lessons/binary_pck/kernels/fk/moon_080317.tf',
                'lessons/binary_pck/kernels/spk/de432s.bsp',
                'lessons/binary_pck/kernels/pck/moon_pa_de421_1900-2050.bpc'
                ]
    spice.furnsh(kernels)
    earth_lclf_me = spice.spkpos(targ='earth', et=et, ref='moon_me', abcorr='CN+S', obs='moon')[0]
    radius, lon, lat = spice.reclat(rectan=earth_lclf_me)
    lon = np.rad2deg(lon)
    lat = np.rad2deg(lat)
    print(f'moon_me:  radius={radius:.6f}km, lon={lon:.6f} deg, lat={lat:.6f} deg')
    spice.kclear()

    #4.
    earthhat_iau = earth_lclf_iau/norm(earth_lclf_iau)
    earthhat_me = earth_lclf_me/norm(earth_lclf_me)
    vsep = np.rad2deg(spice.vsep(v1=earthhat_iau, v2=earthhat_me))
    print(f'iau_moon/moon_me vsep={vsep:.6f} deg')

    #5.
    kernels = ['lessons/binary_pck/kernels/fk/moon_080317.tf',
                'lessons/binary_pck/kernels/spk/de432s.bsp',
                'lessons/binary_pck/kernels/pck/moon_pa_de421_1900-2050.bpc'
                ]
    spice.furnsh(kernels)
    earth_lclf_pa = spice.spkpos(targ='earth', et=et, ref='moon_pa', 
                                 abcorr='CN+S', obs='moon')[0]
    radius, lon, lat = spice.reclat(rectan=earth_lclf_pa)
    lon = np.rad2deg(lon)
    lat = np.rad2deg(lat)
    spice.kclear()
    print(f'moon_pa: radius={radius:.6f}km, lon={lon:.6f} deg, lat={lat:.6f} deg')

    #6.
    earthhat_pa = earth_lclf_pa/norm(earth_lclf_pa)
    vsep = np.rad2deg(spice.vsep(v1=earthhat_pa, v2=earthhat_me))
    print(f'moon_pa/moon_me vsep={vsep:.6f} deg')

    #7.
    kernels = ['lessons/binary_pck/kernels/fk/moon_080317.tf',
                'lessons/binary_pck/kernels/spk/de432s.bsp',
                'lessons/binary_pck/kernels/pck/moon_pa_de421_1900-2050.bpc',
                'lessons/binary_pck/kernels/pck/pck00010.tpc'
                ]
    spice.furnsh(kernels)
    spoint_me, tepch, srfvec,  = spice.subpnt(method='near point: ellipsoid',
                              target='moon', et=et, fixref='moon_me', 
                              abcorr='lt+s', obsrvr='earth')
    sp_radius, sp_lon, sp_lat = spice.reclat(rectan=spoint_me)
    sp_lon = np.rad2deg(sp_lon)
    sp_lat = np.rad2deg(sp_lat)
    print(f'subpoint-moon_me: radius={sp_radius:.6f}km, lon={sp_lon:.6f} deg, lat={sp_lat:.6f} deg')
    spice.kclear()

    #8.
    kernels = ['lessons/binary_pck/kernels/fk/moon_080317.tf',
                'lessons/binary_pck/kernels/spk/de432s.bsp',
                'lessons/binary_pck/kernels/pck/moon_pa_de421_1900-2050.bpc',
                'lessons/binary_pck/kernels/pck/pck00010.tpc'
                ]
    spice.furnsh(kernels)
    spoint_pa, tepch, srfvec,  = spice.subpnt(method='near point: ellipsoid',
                              target='moon', et=et, fixref='moon_pa', 
                              abcorr='lt+s', obsrvr='earth')
    sp_radius, sp_lon, sp_lat = spice.reclat(rectan=spoint_pa)
    sp_lon = np.rad2deg(sp_lon)
    sp_lat = np.rad2deg(sp_lat)
    print(f'subpoint-moon_pa: radius={sp_radius:.6f}km, lon={sp_lon:.6f} deg, lat={sp_lat:.6f} deg')
    spice.kclear()

    #9.
    dist = spice.vdist(v1=spoint_me, v2=spoint_pa)
    print(f'spoint dist sep={dist:.6f}km\n')




    ## Earth Rotation ##

    #1 
    spice.furnsh('lessons/binary_pck/kernels/lsk/naif0012.tls')
    time = spice.utc2et('2020 Apr 05 10:00:00 UTC')
    print(f'Earth Rotation: et, s={time}')
    
    #2.
    kernels = ['lessons/binary_pck/kernels/spk/de432s.bsp',
                'lessons/binary_pck/kernels/pck/pck00010.tpc'
                ]
    spice.furnsh(kernels)
    moon_ecef_iau = spice.spkpos(targ='moon', et=et, ref='iau_earth', abcorr='LT+S', obs='earth')[0]
    radius, lon, lat = spice.reclat(rectan=moon_ecef_iau)
    lon = np.rad2deg(lon)
    lat = np.rad2deg(lat)
    print(f'iau_earth: radius={radius:.6f}km, lon={lon:.6f} deg, lat={lat:.6f} deg')
    spice.kclear()

    #3.
    kernels = ['lessons/binary_pck/kernels/spk/de432s.bsp',
                'lessons/binary_pck/kernels/pck/earth_070425_370426_predict.bpc'
                ]
    spice.furnsh(kernels)
    moon_ecef_itrf = spice.spkpos(targ='moon', et=et, ref='itrf93', abcorr='LT+S', obs='earth')[0]
    radius, lon, lat = spice.reclat(rectan=moon_ecef_itrf)
    lon = np.rad2deg(lon)
    lat = np.rad2deg(lat)
    print(f'earth_itrf:  radius={radius:.6f}km, lon={lon:.6f} deg, lat={lat:.6f} deg')
    spice.kclear()

    #4.
    moonhat_iau = moon_ecef_iau/norm(moon_ecef_iau)
    moonhat_me = moon_ecef_itrf/norm(moon_ecef_itrf)
    vsep = np.rad2deg(spice.vsep(v1=moonhat_iau, v2=moonhat_me))
    print(f'iau_earth/itrf93 vsep={vsep:.6f} deg')

    
    ## determine cause of angular offsets ##
    #5 
    spice.furnsh('lessons/binary_pck/kernels/lsk/naif0012.tls')
    timestring = "2021-APR-05 16:00.000 (UTC)"
    et = spice.utc2et(timestring)
    et100 = et + 100*24*3600.0

    #6-10
    kernels = ['lessons/binary_pck/kernels/spk/de432s.bsp',
                'lessons/binary_pck/kernels/pck/pck00010.tpc',
                'lessons/binary_pck/kernels/pck/earth_070425_370426_predict.bpc']
    spice.furnsh(kernels)
    rmat = spice.pxform(fromstr='iau_earth', tostr='itrf93', et=et)
    print(rmat)
    x_itrf = rmat[0] # x-axis of the itrf93 frame expressed relative to iau_earth frame
    z_itrf = rmat[2] # z-axis of the itrf93 frame expressed relative to iau_earth frame
    x_vsep = np.rad2deg(spice.vsep(v1=x_itrf, v2=[1, 0, 0]))
    print(f'x_itrf - iau_earth x-axis sep = {x_vsep: 0.6f} deg, et')
    z_vsep = np.rad2deg(spice.vsep(v1=z_itrf, v2=[0, 0, 1]))
    print(f'z_itrf - iau_earth z-axis sep = {x_vsep: 0.6f} deg, et')
    # et + 100 days
    rmat = spice.pxform(fromstr='iau_earth', tostr='itrf93', et=et100)
    print(rmat)
    x_itrf = rmat[0] # x-axis of the itrf93 frame expressed relative to iau_earth frame
    z_itrf = rmat[2] # z-axis of the itrf93 frame expressed relative to iau_earth frame
    x_vsep = np.rad2deg(spice.vsep(v1=x_itrf, v2=[1, 0, 0]))
    print(f'x_itrf - iau_earth x-axis sep = {x_vsep: 0.6f} deg, et+100d')
    z_vsep = np.rad2deg(spice.vsep(v1=z_itrf, v2=[0, 0, 1]))
    print(f'z_itrf - iau_earth z-axis sep = {x_vsep: 0.6f} deg, et+100d')
    spice.kclear()

    #11-13
    kernels = ['lessons/binary_pck/kernels/spk/de432s.bsp',
                'lessons/binary_pck/kernels/pck/earth_070425_370426_predict.bpc',
                'lessons/binary_pck/kernels/spk/earthstns_itrf93_050714.bsp',
                'lessons/binary_pck/kernels/fk/earth_topo_050714.tf']
    spice.furnsh(kernels)
    r_moon_dss13_topo = spice.spkpos(targ='moon', et=et, ref='DSS-13_TOPO', abcorr='LT+S', obs='DSS-13')[0]
    radius, lon, lat = spice.reclat(rectan=r_moon_dss13_topo)
    az = -lon
    while az < 0.0:
        az += 2*np.pi
    az_moon_dss13 = -np.rad2deg(az)
    el_moon_dss13 = np.rad2deg(lat)
    print(f'DSS13-Moon Azimuth = {az_moon_dss13: 0.6f} deg, Elevation = {el_moon_dss13: 0.6f} deg')
    spice.kclear()

    #14-15
    kernels = ['lessons/binary_pck/kernels/spk/de432s.bsp',
                'lessons/binary_pck/kernels/pck/earth_070425_370426_predict.bpc',
                'lessons/binary_pck/kernels/pck/pck00010.tpc',
                'lessons/binary_pck/kernels/fk/moon_080317.tf',
                'lessons/binary_pck/kernels/spk/de432s.bsp',
                'lessons/binary_pck/kernels/pck/moon_pa_de421_1900-2050.bpc']
    spice.furnsh(kernels)
    spoint_iau, trgepc, svec = spice.subslr(method='near point: ellipsoid', target='earth', et=et, 
                                        fixref='iau_earth', abcorr='LT+S', obsrvr='sun')
    spoint_dist, spoint_lon, spoint_lat = spice.reclat(rectan=spoint_iau)
    print(f'iau_earth subsolor point: lon = {np.rad2deg(spoint_lon):0.6f} deg, lat = {np.rad2deg(spoint_lat):0.6f} deg')
    spoint_itrf, trgepc, svec = spice.subslr(method='near point: ellipsoid', target='earth', et=et, 
                                        fixref='itrf93', abcorr='LT+S', obsrvr='sun')
    spoint_dist, spoint_lon, spoint_lat = spice.reclat(rectan=spoint_itrf)
    print(f'itrf93 subsolor point: lon = {np.rad2deg(spoint_lon):0.6f} deg, lat = {np.rad2deg(spoint_lat):0.6f} deg')

    #16
    dist = spice.vdist(v1=spoint_iau, v2=spoint_itrf)
    print(f'distance between subsolar points: {dist:0.6f} km')

