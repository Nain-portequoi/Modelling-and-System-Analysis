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
theta = 0.1
psi = 1
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
    lambdA = 0
    nu = 1
    sigma = 0.5
    alpha = 1
    delta = 1
    eta = 1

    pi = np.pi

    # Ecriture des 4 équations différentielles
    dy1_dtau = y2
    dy2_dtau = fonction_epsilon(sigma,y3, xi, alpha, y1) * 4 * pi**2 * delta * eta/sigma * (y3 * np.cos(y1) - alpha * np.sin(y1)) - 4 * pi**2 * np.sin(y1)
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
    method='RK45', 
    rtol=précision_relative, 
    atol=précision_absolue
)



temps = solution.t
y1_sol = solution.y[0]
y2_sol = solution.y[1]
y3_sol = solution.y[2]
y4_sol = solution.y[3]


# Graphique : theta et psi en fonction de tau
plt.figure(figsize=(7, 5))

plt.plot(temps, y1_sol, label=r"$\theta(\tau)$", color="tab:blue")
plt.plot(temps, y3_sol, label=r"$\psi(\tau)$", color="tab:orange")

plt.xlabel(r"$\tau$")
plt.ylabel("Valeurs de " + r"$\theta$" + " et " + r"$\psi$")
plt.title(r"Évolution de $\theta$ et $\psi$ en fonction de $\tau$")
plt.legend(loc="upper left")

plt.tight_layout()


plt.savefig("resultats_simulation.png"); plt.show()   



# obligé de mettre ca à cause de ubuntu mdr :)
