# gravitational coefficients, km3/s2
planetary_mu = {'sun': 1.32712440018e11,
                'mercury': 2.2032e13,
                'venus': 3.24858599e5,
                'earth': 3.986004415e5,
                'mars': 4.28283100e4,
                'jupiter': 1.266865361e8,
                'saturn': 3.7931208e7,
                'uranus': 5.7939513e6,
                'neptune': 6.835100e6,
                'pluto': 8.71e2}

mu_sun = planetary_mu['sun']
mu_venus = planetary_mu['venus']
mu_earth = planetary_mu['earth']
mu_mars = planetary_mu['mars']
mu_jupiter = planetary_mu['jupiter']

# universal gravitational constant
G_univ = 6.67408e-20 # km3/(kg)

# periods, days
planetary_period = {'mercury': None,
                    'venus': 224.701,
                    'earth': 365.242189,
                    'moon': None,
                    'mars': None,
                    'jupiter': None,
                    'saturn': None,
                    'uranus': None,
                    'neptune': None,
                    'pluto': None,}
P_earth = 365.242189 # days
P_venus = 224.701 # days
AU = 1.49597870691e8 # km

# planetary radii, km
planetary_radii = {'mercury': None,
                   'venus': 6051.8,
                   'earth': 6378.14,
                   'moon': 1737.4,
                   'mars': 3396.19,
                   'jupiter': 71492,
                   'saturn': 60268,
                   'uranus': 25559,
                   'neptune': 24764,
                   'pluto': 1188.3}
r_venus = 6051.8
r_earth = 6378.14
r_mars = 3396.19
r_jupiter = 71492

# planetary heliocentric semi-major axes (km)
planetary_sma = {'mercury': 57.909e6,
                 'venus': 0.723 * AU, # from wikipedia
                 'earth': 149598023,
                 'moon': 384400, # from earth
                 'mars': 227939186, # 1.523679342 AU
                 'jupiter': 778.570e6, # from nssdc.gsfc.nasa.gov/planetary/factsheet/
                 'saturn': 1433.529e6,
                 'uranus': 2872.463e6,
                 'neptune': 4495.060e6,
                 'pluto': 5869.656e6}

sma_earth = 149598023
sma_mars = 227939186 # 1.523679342 AU
sma_mars = 1.523679342 * AU
sma_jupiter = 778.570e6 # from nssdc.gsfc.nasa.gov/planetary/factsheet/

# earth constants
E_earth = 0.081819221456
omega_earth = 7.292115e-5 # +/- 1.5e-12 (rad/s)


# imports
import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
import pandas as pd

# shortcut functions
sin, cos, tan = np.sin, np.cos, np.tan
arcsin, arccos, arctan, arctan2 = np.arcsin, np.arccos, np.arctan, np.arctan2
sinh, cosh = np.sinh, np.cosh
arcsinh, arccosh = np.arcsinh, np.arccosh
sqrt = np.sqrt
pi = np.pi
r2d = np.rad2deg
d2r = np.deg2rad


def get_mu(center='earth'):
    try:
        return planetary_mu[f'{center.lower()}']
    except:
        raise ValueError("Undefined planet, ending..")


def get_sma(center='earth'):
    try:
        return planetary_sma[f'{center.lower()}']
    except:
        raise ValueError("Undefined planet, ending..")


def get_period(center='earth'):
    # determine which planet center to compute
    try:
        return planetary_period[f'{center.lower()}']
    except:
        raise ValueError("Undefined planet, ending..")


def get_radius(center='earth'):
    # determine which planet center to compute
    try:
        return planetary_radii[f'{center.lower()}']
    except:
        raise ValueError("Undefined planet, ending..")


if __name__ == '__main__':
    pass
    