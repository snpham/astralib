#! python3
import spiceypy as spice
import numpy as np
from numpy.linalg import norm
import spiceypy.utils.support_types as stypes



if __name__ == "__main__":

    spice.furnsh(['lessons/event_finding/kernels/spk/de405xs.bsp',
                  'lessons/event_finding/kernels/spk/earthstns_itrf93_050714.bsp',
                  'lessons/event_finding/kernels/fk/earth_topo_050714.tf',
                  'lessons/event_finding/kernels/pck/earth_000101_060525_060303.bpc',
                  'lessons/event_finding/kernels/lsk/naif0008.tls',
                  'lessons/event_finding/kernels/spk/ORMM__040501000000_00076XS.BSP',
                  'lessons/event_finding/kernels/pck/pck00008.tpc'])
    
    tdbfmt = 'YYYY MON DD HR:MN:SC.### (TDB) ::TDB'
    maxivl = 1000
    maxwin = 2*maxivl

    srfpt  = 'DSS-14'
    obsfrm = 'DSS-14_TOPO'
    target = 'MEX'
    abcorr = 'CN+S'
    start  = '2004 MAY 2 TDB'
    stop   = '2004 MAY 6 TDB'
    elvlim =  6.0
    
    revlim = np.deg2rad(elvlim)
    crdsys = 'latitudinal'
    coord = 'latitude'
    relate = '>'
    adjust = 0.0
    stepsize = 300.0
    stepsz = stepsize
    et_start = spice.str2et(start)
    et_end = spice.str2et(stop)

    timestr = spice.timout(et=et_start, pictur=tdbfmt)
    print(f'Start time: {timestr}')
    timestr = spice.timout(et=et_end, pictur=tdbfmt)
    print(f'Start time: {timestr}')    
    cnfine = stypes.SPICEDOUBLE_CELL(2)
    spice.wninsd(et_start, et_end, cnfine)
    riswin = stypes.SPICEDOUBLE_CELL(maxwin)
    spice.gfposc(target='MEX', inframe='DSS-14_TOPO', abcorr='lt+s', 
                 obsrvr='DSS-14', crdsys=crdsys, coord='latitude', 
                 relate='>', refval=revlim, adjust=0.0, step=stepsz, 
                 nintvals=maxivl, cnfine=cnfine, result=riswin)
    winsiz = spice.wncard(riswin)
    if winsiz == 0:
        print('No events found')
    else:
        print(f'Visibility times of {target} as seen from {srfpt}')
        for i in range(winsiz):
            intbeg, intend = spice.wnfetd(window=riswin, n=i)
            timestr = spice.timout(et=intbeg, pictur=tdbfmt)
            if i == 0:
                print(f'Visibility or window start time: {timestr}')
            else:
                print(f'Visibility start time: {timestr}')
            timstr = spice.timout(et=intend, pictur=tdbfmt)
            if i == (winsiz-1):
                print(f'Visibility or window stop time: {timestr}')
            else:
                print(f'Visibility stop time: {timestr}')

    # occultations
    spice.furnsh(['lessons/event_finding/kernels/spk/de405xs.bsp',
                  'lessons/event_finding/kernels/spk/earthstns_itrf93_050714.bsp',
                  'lessons/event_finding/kernels/fk/earth_topo_050714.tf',
                  'lessons/event_finding/kernels/pck/earth_000101_060525_060303.bpc',
                  'lessons/event_finding/kernels/lsk/naif0008.tls',
                  'lessons/event_finding/kernels/spk/ORMM__040501000000_00076XS.BSP',
                  'lessons/event_finding/kernels/pck/pck00008.tpc',
                  'lessons/event_finding/kernels/dsk/mars_lowres.bds'])
    back = target
    bshape = 'point'
    bframe = ' '
    front = 'mars'
    fshape = 'ELLIPSOID'
    fframe = 'iau_mars'
    occtyp = 'any'
    
    # cnfine = stypes.SPICEDOUBLE_CELL(2)
    # spice.wninsd(left=et_start, right=et_end, window=cnfine)
    # riswin = stypes.SPICEDOUBLE_CELL(maxwin)
    # spice.gfposc(target='MEX', inframe='DSS-14_TOPO', abcorr='lt+s', 
    #              obsrvr='DSS-14', crdsys=crdsys, coord='latitude', 
    #              relate='>', refval=revlim, adjust=0.0, step=stepsz, 
    #              nintvals=maxivl, cnfine=cnfine, result=riswin)

    print('\nSearing using ellipsoid target shape model...')
    eocwin = stypes.SPICEDOUBLE_CELL(maxwin)
    fshape = 'ELLIPSOID'
    spice.gfoclt(occtyp='any', front='mars', fshape='ELLIPSOID', 
                 fframe='iau_mars', back=target, bshape='point', 
                 bframe=' ', abcorr='CN+S', obsrvr='DSS-14', 
                 step=stepsz, cnfine=riswin, result=eocwin)
    evswin = spice.wndifd(a=riswin, b=eocwin)
    print('Done')

    print('Searching using DSK target shape model...')
    docwin = stypes.SPICEDOUBLE_CELL(maxwin)
    fshape = 'DSK/UNPRIORITIZED'
    spice.gfoclt(occtyp='any', front='mars', fshape='ELLIPSOID', 
                 fframe='iau_mars', back=target, bshape='point', 
                 bframe=' ', abcorr='CN+S', obsrvr='DSS-14', 
                 step=stepsz, cnfine=riswin, result=docwin)
    dvswin = spice.wndifd(a=riswin, b=docwin)
    print('Done\n')

    winsiz = spice.wncard(window=evswin)
    if winsiz == 0:
        print('No events were found')
    else:
        print(f'Visibility start/stop times of {target} as seen from {srfpt} using both ellipsoidal and DSK target shape models\n')
        for i in range(winsiz):
            intbeg, intend = spice.wnfetd(window=evswin, n=i)
            btmstr = spice.timout(et=intbeg, pictur=tdbfmt)
            etmstr = spice.timout(et=intend, pictur=tdbfmt)
            print(f'Ell: {btmstr} : {etmstr}')
            dintbg, dinten = spice.wnfetd(window=dvswin, n=i)
            btmstr = spice.timout(et=dintbg, pictur=tdbfmt)
            etmstr = spice.timout(et=dinten, pictur=tdbfmt)
            print(f'DSK: {btmstr} : {etmstr}')
            
    # spiceypy.gfposc
    # spiceypy.gfdist