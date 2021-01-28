
# earth constants
REq_earth = 6378.1363 # (km)
RPolar_earth = 6356.7516005 # (km) 
F_earth = 0.003352813178 # = 1/298.257
E_earth = 0.081819221456
omega_earth = 7.292115e-5 # +/- 1.5e-12 (rad/s)

# Gravitational coefficients, km3/s2
mu_sun = 1.32712440018e11
mu_venus = 3.24858599e5
mu_earth = 3.98600433e5
mu_mars = 4.28283100e4
mu_jupiter = 1.266865361e8
mu_saturn = 3.7931208e7
mu_uranus = 5.7939513e6
mu_neptune = 6.835100e6
mu_pluto = 8.71e2

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
    