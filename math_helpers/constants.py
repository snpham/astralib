# gravitational coefficients, km3/s2
mu_sun = 1.32712440018e11
mu_mercury = 2.2032e13
mu_venus = 3.24858599e5
mu_earth = 3.986004415e5
mu_mars = 4.28283100e4
mu_jupiter = 1.266865361e8
mu_saturn = 3.7931208e7
mu_uranus = 5.7939513e6
mu_neptune = 6.835100e6
mu_pluto = 8.71e2

# universal gravitational constant
G_univ = 6.67408e-20 # km3/(kg)

# conversion
P_earth = 365.242189 # days
P_venus = 224.701 # days
AU = 1.49597870691e8 # km

# planetary radii, km
r_venus = 6051.8
r_earth = 6378.14
r_mars = 3396.19
r_jupiter = 71492
r_saturn = 60268
r_uranus = 25559
r_neptune = 24764
r_pluto = 1188.3

# planetary heliocentric semi-major axes
sma_earth = 149598023
sma_moon = 384400
sma_mars = 227939186 # 1.523679342 AU
sma_mars = 1.523679342 * AU
sma_venus = 0.723 * AU # from wikipedia
sma_jupiter = 778.570e6 # from nssdc.gsfc.nasa.gov/planetary/factsheet/
sma_saturn = 1433.529e6
sma_uranus = 2872.463e6
sma_neptune = 4495.060e6
sma_pluto = 5869.656e6
sma_mercury = 57.909e6

# earth constants
E_earth = 0.081819221456
omega_earth = 7.292115e-5 # +/- 1.5e-12 (rad/s)


# naming shortcuts
import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
import pandas as pd

sin, cos, tan = np.sin, np.cos, np.tan
arcsin, arccos, arctan, arctan2 = np.arcsin, np.arccos, np.arctan, np.arctan2
sinh, cosh = np.sinh, np.cosh
arcsinh, arccosh = np.arcsinh, np.arccosh
sqrt = np.sqrt
pi = np.pi
r2d = np.rad2deg
d2r = np.deg2rad

def get_mu(center='earth'):
    # determine which planet center to compute
    if center.lower() == 'earth':
        return mu_earth
    elif center.lower() == 'mercury':
        return mu_mercury
    elif center.lower() == 'mars':
        return mu_mars
    elif center.lower() == 'sun':
        return mu_sun
    elif center.lower() == 'venus':
        return mu_venus
    elif center.lower() == 'jupiter':
        return mu_jupiter
    elif center.lower() == 'saturn':
        return mu_saturn
    elif center.lower() == 'uranus':
        return mu_uranus
    elif center.lower() == 'neptune':
        return mu_neptune
    elif center.lower() == 'pluto':
        return mu_pluto
    else:
        raise ValueError("Undefined planet, ending..")


def get_sma(center='earth'):
    # determine which planet center to compute
    if center.lower() == 'earth':
        return sma_earth
    elif center.lower() == 'mercury':
        return sma_mercury
    elif center.lower() == 'mars':
        return sma_mars
    elif center.lower() == 'venus':
        return sma_venus
    elif center.lower() == 'jupiter':
        return sma_jupiter
    elif center.lower() == 'saturn':
        return sma_saturn
    elif center.lower() == 'uranus':
        return sma_uranus
    elif center.lower() == 'neptune':
        return sma_neptune
    elif center.lower() == 'pluto':
        return sma_pluto
    else:
        raise ValueError("Undefined planet, ending..")

def get_period(center='earth'):
    # determine which planet center to compute
    if center.lower() == 'earth':
        return P_earth
    elif center.lower() == 'mercury':
        return None
    elif center.lower() == 'mars':
        return None
    elif center.lower() == 'venus':
        return P_venus
    elif center.lower() == 'jupiter':
        return None
    elif center.lower() == 'saturn':
        return None
    elif center.lower() == 'uranus':
        return None
    elif center.lower() == 'neptune':
        return None
    elif center.lower() == 'pluto':
        return None
    else:
        raise ValueError("Undefined planet, ending..")

def get_radius(center='earth'):
    # determine which planet center to compute
    if center.lower() == 'earth':
        return r_earth
    elif center.lower() == 'mercury':
        return None
    elif center.lower() == 'mars':
        return r_mars
    elif center.lower() == 'venus':
        return r_venus
    elif center.lower() == 'jupiter':
        return r_jupiter
    elif center.lower() == 'saturn':
        return r_saturn
    elif center.lower() == 'uranus':
        return r_uranus
    elif center.lower() == 'neptune':
        return r_neptune
    elif center.lower() == 'pluto':
        return r_pluto
    else:
        raise ValueError("Undefined planet, ending..")


if __name__ == '__main__':
    pass
    