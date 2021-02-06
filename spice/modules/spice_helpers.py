
import os
import pathlib
import spiceypy as sp
import spiceypy.utils.support_types as stypes


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

    max_windows=50
    spk_filename = f'{dir}/kernels/spacecraft/{kernel}.bsp'
    coverage = stypes.SPICEDOUBLE_CELL(max_windows * 2)
    sp.spkcov(spk_filename, id_obj, coverage)
    UTCSTR = sp.et2utc(coverage[0], 'C', 3, 30)
    UTCSTR2 = sp.et2utc(coverage[-1], 'C', 3, 30)
    print(f'\n###>> mission start {UTCSTR}') 
    print(f'###>> mission end {UTCSTR2}\n')
    return sp.wnfetd(coverage, 0)  # mission start and end epoch



if __name__ == '__main__':

    print(os.path.dirname(__file__))
    load_planetary_kernels()
    load_mars2020_kernels()
    print(sp.ktotal('ALL'))
    sp.kclear()
