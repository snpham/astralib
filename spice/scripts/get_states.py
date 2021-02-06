import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.spice_helpers import *
from matplotlib import pyplot as plt




if __name__ == '__main__':
    
    print(sp_dir)
    load_planetary_kernels()
    load_mars2020_kernels(ILS='jez')
    # print(sp.ktotal('ALL'))

    # et = sp.utc2et("2021 FEB 18 20:44:52.000")
    # states = sp.spkezr('-168', et, ref='j2000', abcorr='NONE', obs='499')
    et_start, et_end = spk_coverage(kernel='mars2020/m2020_edl_nom_jez_v2', id_obj=-168)

    dt = 3600
    # ets = np.arange(et_start, et_start+3600*24*7, dt)
    dt = 1
    ets = np.arange(et_start, et_end, dt)


    states = []
    for et in ets:
        states.append(sp.spkezr('-168', et, ref='M2020_TOPO', abcorr='NONE', obs='-168900')[0])
    
    states = np.vstack(states)

    # plots
    fig=plt.figure()
    plt.style.use('seaborn')
    ax=fig.add_subplot(211)
    ax2=fig.add_subplot(212)
    ax.plot((ets-ets[0]), states[:,0], label='rx')
    ax.plot((ets-ets[0]), states[:,1], label='ry')
    ax.plot((ets-ets[0]), states[:,2], label='rz')
    ax.axvline((ets[-1]-ets[0])-353, color='r', label='guidance_start')
    ax.axvline((ets[-1]-ets[0])-159, label='parachute_deploy')
    ax.axvline((ets[-1]-ets[0])-85, color='green', label='trn_begin')
    ax.set_xlabel("edl duration (s)")
    ax.set_ylabel("positions wrt landing site [fixed] (km)")
    ax.set_title("mars 2020 perseverance rover edl for jezero crater landing site [fixed NWU topo frame]")
    ax2.plot((ets-ets[0]), states[:,3], label='vx')
    ax2.plot((ets-ets[0]), states[:,4], label='vy')
    ax2.plot((ets-ets[0]), states[:,5], label='vz')
    ax2.axvline((ets[-1]-ets[0])-353,color='r', label='guidance_start')
    ax2.axvline((ets[-1]-ets[0])-159, label='parachute_deploy')
    ax2.axvline((ets[-1]-ets[0])-85, color='green', label='trn_begin')
    ax2.set_xlabel("edl duration (s)")
    ax2.set_ylabel("velocities wrt landing site [fixed] (km/s)")
    ax2.legend()
    ax.legend()
    fig.tight_layout()
    plt.show()
