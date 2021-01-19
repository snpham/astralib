
# earth constants
REq_earth = 6378.1363 # (km)
RPolar_earth = 6356.7516005 # (km) 
F_earth = 0.003352813178 # = 1/298.257
E_earth = 0.081819221456
omega_earth = 7.292115e-5 # +/- 1.5e-12 (rad/s)
mu_earth = 398600.4418 # (km^3*s^-2) gravitational constants
mu_mars = 42828.37 # (km^3*s^-2) mars

def get_mu(center='earth'):
    # determine which planet center to compute
    if center.lower() == 'earth':
        mu = mu_earth
    elif center.lower() == 'mars':
        mu = mu_mars
    else:
        mu = mu_earth
        print('Using earth as center object\n')

    return mu


if __name__ == '__main__':
    pass
    