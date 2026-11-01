# Projet Modelling and System Analysis


#    Rendu :  manim -pql ton_fichier.py Systeme



# Import de librairies
from manim import *
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Précision

précision_relative = 1e-6
précision_absolue = 1e-8

# Durée de la simulation
duréeSimulation = 10

# Conditions initiales 
theta = -0.5
psi = 0.5
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

#plt.show()   

# ---------------------------------------------------------------------------------------------------------------------------


class Systeme(Scene):
    def construct(self):

        class Parametres:
            def __init__(self):
                self.xi = xi
                self.eta = eta
                self.alpha = alpha

                # --- dessin du système (unités Manim) ---
                self.S = 5          # longueur D à l'écran (plus grand = animation plus grande)
                self.d = 0.7        # côté du carré M
                self.r = 0.2        # rayon de la masse m

                self.sim = np.array([0, duréeSimulation, len(temps)])
                self.ressort_1 = np.array([8, 12, 0.5])  # extended_length, nbr_spire, extremity_length
                self.ressort_2 = np.array([8, 12, 0.5])

                # --- centrage sur l'écran ---
                self.shift_system = np.array([0.5, 2.2, 0])

        P = Parametres()

        simulation = results_simulation(P)

        t = ValueTracker(0)
        frame = always_redraw(lambda: plot_system(P, t.get_value() + P.sim[0], simulation))
        self.add(frame)
        self.play(t.animate.set_value(P.sim[1] - P.sim[0]), rate_func=linear, run_time=P.sim[1] - P.sim[0])






def plot_system(P, t, simulation):

    from scipy.interpolate import interp1d

    # Création de l'interpolateur pour chaque ligne de y
    interpolator = [interp1d(simulation.t, y_row, kind='linear', fill_value="extrapolate") for y_row in simulation.y]

    # Interpolation des valeurs pour ce nouveau temps
    y_interp = np.array([interp(t) for interp in interpolator])

    theta = y_interp[0]
    psi = y_interp[2]

    # --- Passage du repère du rapport vers l'écran ---
    # Origine = pivot O ; e_x vers la gauche, e_y vers le bas ; unité = D
    S = P.S
    L = P.xi / P.eta  # longueur du pendule / D  (car eta = a/L et xi = a/D)

    def pos(u, v):
        return np.array([-S * u, -S * v, 0])

    pos_M = pos(psi, P.alpha)                                              # masse M
    pos_wall = pos(1, P.alpha)                                             # mur (à distance D)
    pos_attache = pos(P.xi * np.sin(theta), P.xi * np.cos(theta))          # attache du ressort 2 (a e_r)
    pos_m = pos(L * np.sin(theta), L * np.cos(theta))                      # masse m (L e_r)

    y_sol = pos_M[1] - P.d / 2   # sol sur lequel glisse M
    y_top = 0.6                  # hauteur de la flèche psi

    # --- Objets ---
    wall = Rectangle(height=y_top - y_sol, width=P.d, color=BLUE, fill_opacity=0.5).move_to(np.array([pos_wall[0] - P.d / 2, (y_top + y_sol) / 2, 0]))
    wall_line = Line(start=[pos_wall[0], y_sol, 0], end=[pos_wall[0], y_top, 0], stroke_width=5, color=RED)
    floor = Line(start=[pos_wall[0], y_sol, 0], end=[0.6, y_sol, 0], stroke_width=5)

    ceiling = Line(start=[-0.4, 0, 0], end=[0.4, 0, 0], stroke_width=5)
    dashed = DashedLine([0, 0, 0], [0, y_top, 0], dash_length=0.1)
    dashed_pend = DashedLine([0, 0, 0], [0, -S * L, 0], dash_length=0.1, stroke_width=2)
    dashed_M = DashedLine([pos_M[0], pos_M[1], 0], [pos_M[0], y_top, 0], dash_length=0.1, stroke_width=2)

    rod = Line(ORIGIN, pos_m, stroke_width=3)
    mass_m = Circle(radius=P.r, color=BLUE, fill_opacity=0.5, stroke_color=RED).move_to(pos_m)
    attache = Dot(pos_attache, radius=0.05, color=YELLOW)

    spring_1 = plot_spring(pos_wall[0], pos_wall[1], pos_M[0], pos_M[1], P.ressort_1[0], int(P.ressort_1[1]), P.ressort_1[2])
    spring_2 = plot_spring(pos_M[0], pos_M[1], pos_attache[0], pos_attache[1], P.ressort_2[0], int(P.ressort_2[1]), P.ressort_2[2])
    mass_M = RoundedRectangle(corner_radius=P.d / 5, height=P.d, width=P.d, color=BLUE, fill_opacity=0.5, stroke_color=RED).move_to(pos_M)

    psi_arrow = Arrow([0, y_top, 0], [pos_M[0], y_top, 0], buff=0, stroke_width=3, max_tip_length_to_length_ratio=0.12)
    psi_label = Tex(rf"$\psi(\tau) = {psi:.2f}$", font_size=36).next_to(psi_arrow.get_midpoint(), direction=UP * 0.5)
    theta_label = Tex(rf"$\theta(\tau) = {theta / np.pi * 180:.2f}^\circ$", font_size=36).next_to(np.array([0.6, y_top, 0]), RIGHT)

    return VGroup(wall, wall_line, floor, ceiling, dashed, dashed_pend, dashed_M, rod, spring_1, spring_2, mass_m, mass_M, attache,psi_arrow, psi_label, theta_label).shift(P.shift_system)


# plot_spring : fonction du prof, STRICTEMENT inchangée
def plot_spring(x_1, y_1, x_2, y_2, extended_length, nbr_spire, extremity_length):

    rest_length = np.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)
    theta = np.arctan2(y_2 - y_1, x_2 - x_1)  # Utilisation de arctan2 pour éviter les erreurs

    M = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    c = (rest_length - 2 * extremity_length) / (4 * nbr_spire)
    b = (rest_length - 2 * extended_length) / (4 * nbr_spire)
    d = np.sqrt(b**2 - c**2)

    coord = np.zeros((2, 4 * nbr_spire + 3))
    coord[0, -1] = rest_length

    for i in range(1, 4 * nbr_spire + 2):
        if (i - 1) % 4 == 0 or (i - 3) % 4 == 0:
            coord[0, i] = extremity_length + (i - 1) * c
        if (i - 2) % 4 == 0:
            coord[0, i] = extremity_length + (i - 1) * c
            coord[1, i] = d
        if (i - 4) % 4 == 0:
            coord[0, i] = extremity_length + (i - 1) * c
            coord[1, i] = -d

    coord = M @ coord
    coord[0, :] += x_1
    coord[1, :] += y_1

    group = VGroup()
    for i in range(0, len(coord[0]) - 1):
        line = Line(
            start=[coord[0, i], coord[1, i], 0],
            end=[coord[0, i+1], coord[1, i+1], 0],
            stroke_width=5,
            color=WHITE
        )
        group.add(line)

        # Ajouter un petit cercle aux jonctions pour adoucir l'effet
        dot = Dot([coord[0, i+1], coord[1, i+1], 0], radius=0.025, color=WHITE)
        group.add(dot)

    return group


def results_simulation(P):

    return solution
