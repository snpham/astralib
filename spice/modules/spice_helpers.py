
import os
import pathlib
import spiceypy as sp
import spiceypy.utils.support_types as stypes
import subprocess
import re

sp_dir = pathlib.Path(os.path.abspath(os.path.join(pathlib.Path(__file__).parent.absolute(), '..')))


def load_planetary_kernels():
    
    sp.furnsh([f'{sp_dir}/kernels/solarsystem/de438s.bsp',
               f'{sp_dir}/kernels/solarsystem/naif0012.tls',
               f'{sp_dir}/kernels/solarsystem/pck00010.tpc'
              ])


def load_mars2020_kernels(ILS=None):

    kernels = []
    mars2020_dir = f'{sp_dir}/kernels/spacecraft/mars2020'

    # load all files
    # files = os.listdir(mars2020_dir)
    # for file in files:
    #     kernels.append(f'{mars2020_dir}/{file}')
    # sp.furnsh(kernels)

    files = []
    if ILS == 'clh':
        print(f'loading ils: {ILS}')
        files = ['m2020_cruise_nom_clh_v2.bsp',
                 'm2020_edl_nom_clh_v2.bsp',
                 'm2020_lmst_clh_v2.tsc',
                 'm2020_ls_clh_iau2000_v2.bsp',
                 'm2020_tp_clh_iau2000_v2.tf'
                ]

    if ILS == 'jez':
        print(f'loading ils: {ILS}')
        files = ['m2020_atls_jez_v4.bsp',
                 'm2020_cruise_nom_jez_v2.bsp',
                 'm2020_edl_nom_jez_v2.bsp',
                 'm2020_lmst_jez_v4.tsc',
                 'm2020_ls_jez_iau2000_v4.bsp',
                 'm2020_tp_jez_iau2000_v4.tf'
                ]

    for file in files:
        kernels.append(f'{mars2020_dir}/{file}')

    sp.furnsh(kernels)


def spk_coverage(kernel, id_obj=None, dir=sp_dir):

    max_windows=20
    path = f'{dir}/kernels/spacecraft/{kernel}.bsp'
    if os.path.exists(path):
        spk_path = path
    else:
        spk_path = kernel

    coverage = stypes.SPICEDOUBLE_CELL(max_windows*2)
    sp.spkcov(spk_path, id_obj, coverage)
    UTCSTR = sp.et2utc(coverage[0], 'C', 3, 30)
    UTCSTR2 = sp.et2utc(coverage[-1], 'C', 3, 30)
    # print(f'\n###>>(id {id_obj}) mission start (UTC) {UTCSTR}') 
    # print(f'###>>(id {id_obj}) mission end (UTC) {UTCSTR2}\n')
    return sp.wnfetd(coverage, 0)  # mission start and end epoch


def get_scid(sc_kernel):
    """get spacecraft id
    """
    cmd = ['spicey/brief', sc_kernel]
    result = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode('utf-8')
    sc_id = int(re.search('Body: -[0-9]+', result)[0].split(':')[1])
    return sc_id


if __name__ == '__main__':

    print(os.path.dirname(__file__))
    load_planetary_kernels()
    load_mars2020_kernels()
    print(sp.ktotal('ALL'))
    sp.kclear()
