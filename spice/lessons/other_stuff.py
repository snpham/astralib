#! python3
import spiceypy as spice
import numpy as np
from numpy.linalg import norm
from spiceypy.utils.support_types import SpiceyError

if __name__ == "__main__":

    #1
    meta = 'lessons\other_stuff\kpool.tm'
    spice.furnsh(meta)
    kcount = spice.ktotal('all')
    print(f'Kernel count load: {kcount}')
    for i in range(0, kcount):
        file, type, source, handle = spice.kdata(which=i, kind='all')
        print(f'file: {file}, type: {type}, source: {source}, handle: {handle} ')
    spice.unload('lessons/other_stuff/kernels/spk/de405s.bsp')
    kcount2 = spice.ktotal('all')
    print(f'Kernel count load after unloading 1: {kcount2}')
    spice.unload(meta)
    kcount3 = spice.ktotal('all')
    print(f'Kernel count load after unloading META: {kcount3}\n')
    spice.kclear

    #2.
    nitems = 20
    spice.furnsh('lessons\other_stuff\kervar.tm')
    start = 0
    tmplate = '*RING*'
    try:
        cvals = spice.gnpool(name=tmplate, start=start, room=nitems)
        print(f'Number variables matching template: {len(cvals)}')
    except SpiceyError:
        print( 'No kernel variables matched template.' )
        exit()
    
    for cval in cvals:
        dim, type = spice.dtpool(cval)
        print(f'Number items: {dim}, Of type: {type}')

        if type == 'N':
            dvars = spice.gdpool(name=cval, start=start, room=nitems)
            for dvars in dvars:
                print(f' Numeric value {dvars:0.6f}')
        elif type == 'C':
            cvars = spice.gcpool(name=cval, start=start, room=nitems)
            for cvar in cvars:
                print(f'String value: {cvar}')
        else:
            print('Unkonwn type error')
    dvars = spice.gdpool(name='EXAMPLE_TIMES', start=start, room=nitems)
    print('EXAMPLE_TIMES')
    for dvar in dvars:
        print(f'Time value: {dvar:0.6f}')

    #3.
    spice.furnsh(['lessons\other_stuff\kernels\lsk\\naif0008.tls',
    'lessons\other_stuff\kernels\pck\pck00008.tpc',
    'lessons\other_stuff\kernels\spk\de432s.bsp'])
    et_str = '2020 Apr 12 00:00:00'
    et = spice.utc2et(et_str)
    
    ra, rc = spice.bodvcd(bodyid=399, item='RADII', maxn=3)
    f = (ra - rc) / ra

    r_earth2moon = spice.spkpos(targ='moon', et=et, ref='j2000', abcorr='LT+S', obs='earth')[0]
    print(f'Time: {et_str}, inertial = {r_earth2moon} km')
    rho, ra, dec = spice.recrad(rectan=r_earth2moon)
    print(f'Range/Ra/Dec: Range: {rho:0.04f}km, ra: {np.rad2deg(ra):0.04f}deg, dec: {np.rad2deg(dec):0.4f}deg')
    rho, lon, lat = spice.reclat(r_earth2moon)
    print(f'Latitudinal: Range: {rho:0.04f}km, lon: {np.rad2deg(lon):0.04f}deg, lat: {np.rad2deg(lat):0.4f}deg')
    rho, colat, lon = spice.recsph(rectan=r_earth2moon)
    print(f'Spherical: Range: {rho:0.04f}km, lon: {np.rad2deg(lon):0.04f}deg, colat: {np.rad2deg(colat):0.4f}deg')

    r_earth2moon = spice.spkpos(targ='moon', et=et, ref='iau_earth', abcorr='LT+S', obs='earth')[0]
    print(f'Time: {et_str}, body-fixed = {r_earth2moon} km')
    rho, ra, dec = spice.recrad(rectan=r_earth2moon)
    print(f'Range/Ra/Dec: Range: {rho:0.04f}km, ra: {np.rad2deg(ra):0.04f}deg, dec: {np.rad2deg(dec):0.4f}deg')
    rho, lon, lat = spice.reclat(r_earth2moon)
    print(f'Latitudinal: Range: {rho:0.04f}km, lon: {np.rad2deg(lon):0.04f}deg, lat: {np.rad2deg(lat):0.4f}deg')
    rho, colat, lon = spice.recsph(rectan=r_earth2moon)
    print(f'Spherical: Range: {rho:0.04f}km, lon: {np.rad2deg(lon):0.04f}deg, colat: {np.rad2deg(colat):0.4f}deg')   

    #4

    #5
    SPICETRUE = True
    SPICEFALSE = False
    doloop = SPICETRUE
    doloop = SPICEFALSE

    while(doloop):
        targ = input('Target: ')

        if targ == 'none':
            doloop = SPICEFALSE
        else:
            try:
                state = spice.spkezr(targ=targ, et= 0., ref='j2000', abcorr='LT+S', obs='earth')[0]
                print(f'{targ}, eci [x,y,z]    = {state[0:3]} km')
                print(f'{targ}, eci [vx,vy,zz] = {state[3:6]} km/s')
            except SpiceyError as err:
                print(err)

    #6
    MAXSIZ = 8

    los = [ 'Jan 1, 2003 22:15:02', 'Jan 2, 2003  4:43:29',
            'Jan 4, 2003  9:55:30', 'Jan 4, 2003 11:26:52',
            'Jan 5, 2003 11:09:17', 'Jan 5, 2003 13:00:41',
            'Jan 6, 2003 00:08:13', 'Jan 6, 2003  2:18:01' ]

    phase = [ 'Jan 2, 2003 00:03:30', 'Jan 2, 2003 19:00:00',
            'Jan 3, 2003  8:00:00', 'Jan 3, 2003  9:50:00',
            'Jan 5, 2003 12:00:00', 'Jan 5, 2003 12:45:00',
            'Jan 6, 2003 00:30:00', 'Jan 6, 2003 23:00:00' ]

    los_et = spice.str2et(los)
    phs_et = spice.str2et(phase)
    
    loswin = spice.stypes.SPICEDOUBLE_CELL( MAXSIZ )
    phswin = spice.stypes.SPICEDOUBLE_CELL( MAXSIZ )

    for i in range(0, int(MAXSIZ/2)):
        spice.wninsd(left=los_et[2*i], right=los_et[2*i+1], window=loswin)
        spice.wninsd(left=phs_et[2*i], right=phs_et[2*i+1], window=phswin)

    spice.wnvald(insize=MAXSIZ, n=MAXSIZ, window=loswin)
    spice.wnvald(insize=MAXSIZ, n=MAXSIZ, window=phswin)

    sched = spice.wnintd(a=loswin, b=phswin)
    print(f'Number data values in sched: {spice.card(sched)}')
    print(f'Time intervals meeting the defined critterion')

    for i in range(spice.card(sched)//2):
        left, right = spice.wnfetd(window=sched, n=i)
        utcstr_l = spice.et2utc(left, 'c' , 3)
        utcstr_r = spice.et2utc(right, 'c', 3)
        print(f'{i}, {utcstr_l}    {utcstr_r}')
    
    meas, avg, stddev, small, large = spice.wnsumd(window=sched)
    print(f'Total measure of sched:   {meas:0.6f}\n'
          f'Average measure of sched: {avg:0.6f}\n'
          f'Std dev of meas. in sched: {stddev:0.6f}\n'
          )

    #7
    ufrm = 'km'
    uto = 'm'
    value = 1000
    tvalue = spice.convrt(x=value, inunit=ufrm, outunit=uto)
    print(tvalue)
    clight = spice.clight()
    print(f'speed of light = {clight} km/s')