# Projet Modelling and System Analysis

# Import de librairies
import numpy as np
from scipy.integrate import solve_ivp

# Durée de la simulation
duréeSimulation = 10

# Conditions initiales 
theta = 0
psi = 0.5
vitesseDeRotation = 0
vitesseDeVariabilitéDuRapportDeLongueur = 0

# Définition de la fonction 
def fonction_epsilon(sigma, y3, xi, alpha, y1):
    epsilon = 1 - sigma/np.sqrt(y3**2 + xi**2 + alpha**2 - 2 * xi * y3 * np.sin(y1) - 2 * alpha * xi * np.cos(y1))
    return epsilon

# Définition de la fonction que solve_ivp va utiliser 
def systeme_4_equations(tau, U):
    y1, y2, y3, y4 = U

    # Choix des valeurs de paramètres adimensionnels
    gamma = 1
    mu = 1
    xi = 1
    lambdA = 1
    nu = 0
    sigma = 1
    alpha = 1
    delta = 1
    eta = 1

    pi = np.pi

    # Ecriture des 4 équations différentielles
    dy1_dtau = y2
    dy2_dtau = fonction_epsilon(sigma,y3, xi, alpha, y1) * 4 * pi**2 * delta * eta/sigma * (y3 * np.cos(y1) - alpha * np.sin(y1)) - 4 * pi**2 * np.sin(y1)
    dy3_dtau = y4
    dy4_dtau = 1 / (mu**2) * 4 * pi**2 * (1 - y3 - lambdA) - 1 / (gamma**2) * fonction_epsilon(sigma,y3, xi, alpha, y1) * 4 * pi**2 * (y3 * xi * np.sin(y1)) - nu * y4

    return(dy1_dtau, dy2_dtau, dy3_dtau, dy4_dtau)

# Définition des conditions initiales 
U0 = [theta, vitesseDeRotation, psi, vitesseDeVariabilitéDuRapportDeLongueur]

# Intervalle de temps (durée de la simulation)
t_span = (0, duréeSimulation)

solution = solve_ivp(systeme_4_equations, t_span, U0)

temps = solution.t
y1_sol = solution.y[0]
y2_sol = solution.y[1]
y3_sol = solution.y[2]
y4_sol = solution.y[3]

# test push

