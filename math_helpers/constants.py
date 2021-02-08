# gravitational coefficients, km3/s2
mu_sun = 1.32712440018e11
mu_venus = 3.24858599e5
mu_earth = 3.98600433e5
mu_mars = 4.28283100e4
mu_jupiter = 1.266865361e8
mu_saturn = 3.7931208e7
mu_uranus = 5.7939513e6
mu_neptune = 6.835100e6
mu_pluto = 8.71e2

# conversion
days_per_year = 365.242189
AU = 1.49597870700e8 # km

# planetary radii, km
r_venus = 60513.8
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

# earth constants
E_earth = 0.081819221456
omega_earth = 7.292115e-5 # +/- 1.5e-12 (rad/s)


# naming shortcuts
import numpy as np
from numpy.linalg import norm
sin, cos, tan = np.sin, np.cos, np.tan
arcsin, arccos, arctan = np.arcsin, np.arccos, np.arctan
sinh, cosh = np.sinh, np.cosh
arcsinh, arccosh = np.arcsinh, np.arccosh
sqrt = np.sqrt
pi = np.pi

from matplotlib import pyplot as plt


def get_mu(center='earth'):
    # determine which planet center to compute
    if center.lower() == 'earth':
        return mu_earth
    elif center.lower() == 'mars':
        return mu_mars
    elif center.lower() == 'sun':
        return mu_sun
    else:
        print('Using earth as center object\n')
        return mu_earth


if __name__ == '__main__':
    pass
    