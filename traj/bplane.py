#!/usr/bin/env python3
import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers.constants import *
from traj import conics
from math_helpers.vectors import vcrossv, vdotv, vxs


def get_rp(vinf, psi, mu):
    rp = mu/vinf**2 * ( 1/(cos((pi-psi)/2)) - 1 )
    return rp


def get_turnangle(vinf, rp, mu):
    psi = pi - 2 * arccos(1 / (1 + vinf**2*rp/mu))
    return psi


def bplane_rv(rvec, vvec, v_inf=None, center='sun'):
    """converts trajectory state vectors to B-Plane parameters for a 
    spacecraft approaching a target body.
    tested with hw5 asen6008
    """

    mu = get_mu(center=center)
    # Kep = conics.Keplerian(rvec, vvec, center=center)

    rvec = np.array(rvec)
    vvec = np.array(vvec)

    h_hat = vcrossv(rvec, vvec) / norm(vcrossv(rvec, vvec))
    k_hat = [0, 0, 1]
    vmag = norm(vvec)
    rmag = norm(rvec)
    e_vec = ((vmag**2-mu/rmag)*rvec - vdotv(rvec, vvec)*vvec) / mu

    emag = norm(e_vec)
    e_hat = e_vec/emag
    energy = vmag**2/2 - mu/rmag
    a = - mu/ (2*energy)
    c = a*emag
    # b = sqrt(c**2 - a**2)
    b = abs(a)*sqrt(emag**2-1)

    hxe_hat = vcrossv(h_hat, e_vec) / norm(vcrossv(h_hat, e_vec))
    rho = arccos(1/emag) # asymptote 1/2 angle
    
    S_hat = cos(rho) * e_hat + sin(rho)*hxe_hat
    T_hat = vcrossv(S_hat, k_hat) / norm(vcrossv(S_hat, k_hat))
    R_hat = vcrossv(S_hat, T_hat)

    B_hat = vcrossv(S_hat, h_hat)
    B_vec = vxs(b, B_hat)

    BT = vdotv(B_vec, T_hat)
    BR = vdotv(B_vec, R_hat)
    # print(BR, BT)

    return BT, BR


def bplane_vinf(vinf_in, vinf_out, center='earth', rtn_rp=False):
    """converts trajectory v-infinity vectors to B-Plane parameters for a 
    spacecraft approaching a flyby.
    :param vinf_in: incoming hyperbolic velocity relative to the center (km/s)
    :param vinf_out: outgoing hyperbolic velocity relative to the center (km/s)
    :param center: planet of targetting b-plane
    :return psi: turn angle (rad)
    :return rp: radius of closest approach (km)
    :return BT: BT vector (km)
    :return BR: BR vector (km)
    :return B: magnitude of B vector (km)
    :return theta: angle between the B and T vector (rad)
    """

    mu = get_mu(center=center)

    vmaginf_in = norm(vinf_in)
    vmaginf_out = norm(vinf_out)
    vinf_cross = vcrossv(vinf_in, vinf_out)
    vinf_dot = vdotv(vinf_in, vinf_out)
    S_hat = vinf_in / vmaginf_in
    h_hat = vinf_cross / norm(vinf_cross)
    B_hat = vcrossv(S_hat, h_hat)
    K_hat = [0, 0, 1]
    T_hat = vcrossv(S_hat, K_hat) / norm(vcrossv(S_hat, K_hat))
    R_hat = vcrossv(S_hat, T_hat)

    psi = arccos(vinf_dot / (vmaginf_in*vmaginf_out))
    rp = mu / vmaginf_in**2 * ( 1/(cos((pi-psi)/2)) - 1 )

    if rtn_rp:
        return rp

    B = mu/vmaginf_in**2 * ( ( 1+vmaginf_in**2*rp/mu )**2 - 1 )**(1/2)
    B_vec = vxs(B, B_hat)
    BT = vdotv(B_vec, T_hat)
    BR = vdotv(B_vec, R_hat)
    theta = arccos(vdotv(T_hat, B_hat))
    if vdotv(B_hat, R_hat) < 0:
        theta = 2*pi - theta

    return np.array([psi, rp, BT, BR, B, theta])

    
class BPlane(object):
    """Class to compute Bplane parameters via state vectors or
    inbound/outbound hyperbolic velocities. 
    :param rvec: positional vectors of spacecraft (km)
    :param vvec: velocity vectors of spacecraft (km/s)
    :param vinf_in: incoming hyperbolic velocity relative to the center (km/s)
    :param vinf_out: outgoing hyperbolic velocity relative to the center (km/s)
    :param center: planet of targetting b-plane
    :return psi: turn angle (rad)
    :return rp: radius of closest approach (km)
    :return BT: BT vector (km)
    :return BR: BR vector (km)
    :return B_mag: magnitude of B vector (km)
    :return theta: angle between the B and T vector (rad)
    """

    def __init__(self, rvec=None, vvec=None, vinf_in=None, vinf_out=None, 
                 center='earth'):

        vectors = 'vinf'
        if rvec:
            vectors = 'states'

        self.mu = get_mu(center=center)

        if vectors == 'states':
            """converts trajectory state vectors to B-Plane parameters for a 
            spacecraft approaching a target body.
            tested with hw5 asen6008
            """

            Kep = conics.Keplerian(rvec, vvec, center=center)

            rvec = np.array(rvec)
            vvec = np.array(vvec)
            rxv =  vcrossv(rvec, vvec)

            H_hat = rxv / norm(rxv)
            K_hat = [0, 0, 1]

            # e_vec = ((vmag**2-self.mu/rmag)*rvec - vdotv(rvec, vvec)*vvec) / self.mu
            # assert np.allclose(e_vec, Kep.e_vec)
            # emag = norm(e_vec)
            # e_hat = e_vec/emag
            # assert np.allclose(e_hat, Kep.e_hat)

            # energy = vmag**2/2 - self.mu/rmag
            # assert np.allclose(energy, Kep.energy)
            a = - self.mu/ (2*Kep.energy)
            c = a*Kep.e_mag
            # b = sqrt(c**2 - a**2)
            self.B_mag = abs(a)*sqrt(Kep.e_mag**2-1)

            rho = arccos(1/Kep.e_mag) # asymptote 1/2 angle
            hxe_hat = vcrossv(H_hat, Kep.e_vec) / norm(vcrossv(H_hat, Kep.e_vec))
            self.S_hat = cos(rho) * Kep.e_hat + sin(rho)*hxe_hat
            sxk = vcrossv(self.S_hat, K_hat)

            self.T_hat = sxk / norm(sxk)
            self.R_hat = vcrossv(self.S_hat, self.T_hat)
            self.B_hat = vcrossv(self.S_hat, H_hat)
            self.B_vec = vxs(self.B_mag, self.B_hat)

            self.BdotT = vdotv(self.B_vec, self.T_hat)
            self.BdotR = vdotv(self.B_vec, self.R_hat)

        elif vectors == 'vinf':
            """converts trajectory v-infinity vectors to B-Plane parameters for a 
            spacecraft approaching a flyby.
            tested with Hw 6 asen6008
            """

            self.vmaginf_in = norm(vinf_in)
            self.vmaginf_out = norm(vinf_out)
            vinf_cross = vcrossv(vinf_in, vinf_out)
            vinf_dot = vdotv(vinf_in, vinf_out)

            H_hat = vinf_cross / norm(vinf_cross)
            K_hat = [0, 0, 1]

            self.S_hat = vinf_in / self.vmaginf_in
            sxk = vcrossv(self.S_hat, K_hat)

            self.T_hat = sxk / norm(sxk)
            self.R_hat = vcrossv(self.S_hat, self.T_hat)
            self.B_hat = vcrossv(self.S_hat, H_hat)

            self.psi = arccos(vinf_dot / (self.vmaginf_in*self.vmaginf_out))
            self.rp = self.mu / self.vmaginf_in**2 * ( 1/(cos((pi-self.psi)/2)) - 1 )

            self.B_mag = self.mu/self.vmaginf_in**2 * \
                ( ( 1+self.vmaginf_in**2*self.rp/self.mu )**2 - 1 )**(1/2)
            self.B_vec = vxs(self.B_mag, self.B_hat)
            self.BdotT = vdotv(self.B_vec, self.T_hat)
            self.BdotR = vdotv(self.B_vec, self.R_hat)
            theta = arccos(vdotv(self.T_hat, self.B_hat))
            if vdotv(self.B_hat, self.R_hat) < 0:
                theta = 2*pi - theta
            self.theta = theta

    # @property
    # def rp(self, psi):
    #     return self.mu/vmaginf_in**2 * ( 1/(cos((pi-psi)/2)) - 1 )

    # @property
    # def psi(self, rp):
    #     return pi - 2 * arccos(1 / (1 + vmaginf_in**2*rp/self.mu))
