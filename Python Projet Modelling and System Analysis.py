# Projet Modelling and System Analysis

# Import de librairies
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Précision

précision_relative = 1e-6
précision_absolue = 1e-8

# Durée de la simulation
duréeSimulation = 10

# Conditions initiales 
theta = 1
psi = 0.8
vitesseDeRotation = 0
vitesseDeVariabilitéDuRapportDeLongueur = 0

# Choix des valeurs de paramètres adimensionnels
gamma = 1
mu = 1
xi = 0.5
lambdA = 0.5
nu = 1
sigma = 0.5
alpha = 0.5
delta = 1
eta = 0.5
pi = np.pi

parametres_adim = (gamma, mu, xi, lambdA, nu, sigma, alpha, delta, eta, pi)


# Définition de la fonction 
def fonction_epsilon(sigma, y3, xi, alpha, y1):
    epsilon = 1 - sigma/np.sqrt(y3**2 + xi**2 + alpha**2 - 2 * xi * y3 * np.sin(y1) - 2 * alpha * xi * np.cos(y1))
    return epsilon

# Définition de la fonction que solve_ivp va utiliser 
def systeme_4_equations(tau, U, gamma, mu, xi, lambdA, nu, sigma, alpha, delta, eta, pi):
    y1, y2, y3, y4 = U


    # Ecriture des 4 équations différentielles
    dy1_dtau = y2
    dy2_dtau = 4*pi**2 *(delta * eta / sigma) * fonction_epsilon(sigma, y3, xi, alpha, y1) * (y3 * np.cos(y1) - alpha * np.sin(y1)) - 4 * pi**2 * np.sin(y1)
    dy3_dtau = y4
    dy4_dtau = 1 / (mu**2) * 4 * pi**2 * (1 - y3 - lambdA) - 1 / (gamma**2) * fonction_epsilon(sigma, y3, xi, alpha, y1) * 4 * pi**2 * (y3 - xi * np.sin(y1)) - nu * y4

    return(dy1_dtau, dy2_dtau, dy3_dtau, dy4_dtau)




# Définition des conditions initiales 
U0 = [theta, vitesseDeRotation, psi, vitesseDeVariabilitéDuRapportDeLongueur]

# Intervalle de temps (durée de la simulation)
t_span = (0, duréeSimulation)

solution = solve_ivp(
    systeme_4_equations, 
    t_span, 
    U0,
    args=parametres_adim, 
    method='RK45', 
    rtol=précision_relative, 
    atol=précision_absolue
)


temps = solution.t
y1_sol = solution.y[0]
y2_sol = solution.y[1]
y3_sol = solution.y[2]
y4_sol = solution.y[3]



# ---------------- ENERGIE ---------------------------------

# Calcul de la norme ||r||

norm_r = np.sqrt(y3_sol**2 + xi**2 + alpha**2 - 2 * xi * y3_sol * np.sin(y1_sol) - 2 * xi * alpha * np.cos(y1_sol))


# Energie cinétique 

Ek = (gamma**2 / sigma) * (y4_sol**2) + (xi / (delta * eta)) * (y2_sol**2)

# Energie potentiel 

Ep = (4 * pi**2 * gamma**2 * (1 - y3_sol - lambdA)**2) / (mu**2 * sigma) + (4 * pi**2 * (norm_r - sigma)**2) / sigma - (8 * pi**2 * xi * np.cos(y1_sol)) / (delta * eta) - (8 * pi**2 * gamma**2 * xi * alpha) / (eta * sigma)
Ep_rel = Ep - Ep[0]

# Energie totale

E_totale = Ek + Ep_rel


# Graphique : theta et psi en fonction de thau
plt.figure(1, figsize=(7, 5))

plt.plot(temps, y1_sol, label=r"$\theta(\tau)$", color="tab:blue")
plt.plot(temps, y3_sol, label=r"$\psi(\tau)$", color="tab:orange")

plt.xlabel(r"$\tau$")
plt.ylabel("Valeurs de " + r"$\theta$" + " et " + r"$\psi$")
plt.title(r"Évolution de $\theta$ et $\psi$ en fonction de $\tau$")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("resultats_simulation.png"); 

# Graphique : énergie en fonction de thau 

plt.figure(2, figsize=(7, 5))
plt.plot(temps, E_totale, label="Énergie Totale", color="black")
plt.plot(temps, Ek, label="Énergie cinétique", color="tab:red")
plt.plot(temps, Ep_rel, label="Énergie potentielle", color="tab:green")
plt.xlabel(r"$\tau$")
plt.ylabel("Énergie")
plt.legend()
plt.tight_layout()
plt.savefig("resultats_energie.png")

plt.show()   


