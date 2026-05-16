"""
Module Atmosphérique ISA — International Standard Atmosphere

Calcule les propriétés de l'atmosphère (T, P, ρ, a) et les ratios (θ, δ, σ)
en fonction de l'altitude h [m] et de la déviation de température ΔISA [°C].

Couches modélisées :
  - Troposphère    : h ≤ 11 000 m  — gradient L = −6.5 K/1 000 m
  - Basse Strato.  : h > 11 000 m ~ 20 000 m — température constante (gradient nul)
"""

import numpy as np

# Métadonnées du module (utilisées par main.py pour le menu)
NOM         = "Module Atmosphérique"
DESCRIPTION = "Propriétés T, P, ρ, a et ratios θ, δ, σ en fonction de h et ΔISA"

# ---------------------------------------------------------------------------
# Constantes ISA au niveau au niveau de reference P0
# ---------------------------------------------------------------------------
T0    = 288.15     # Température standard MSL        [K]
P0    = 101_325.0  # Pression standard MSL            [Pa]
RHO0  = 1.225      # Masse volumique standard MSL     [kg/m³]
A0    = 340.294    # Vitesse du son standard MSL      [m/s]

# ---------------------------------------------------------------------------
# Constantes physiques de l'air — supposées constantes sur toute la plage
# ---------------------------------------------------------------------------
G     = 9.80665    # Accélération gravitationnelle    [m/s²]
R     = 287.05     # Constante gaz parfait de l'air   [J/(K·kg)]
GAMMA = 1.4        # Rapport des chaleurs spécifiques [-]

# ---------------------------------------------------------------------------
# Paramètres des couches atmosphériques
# ---------------------------------------------------------------------------
L        = -0.0065   # Gradient thermique troposphère   [K/m]  (−6.5 K/1 000 m)
H_TROPO  = 11_000.0  # Altitude de la tropopause        [m]

# Température à la tropopause en ISA standard (calculée une seule fois)
T11 = T0 + L * H_TROPO  # = 216.65 K

_EXP = -G / (R * L)  # ≈ 5.2561


# ---------------------------------------------------------------------------
# Fonctions de propriétés atmosphériques
# ---------------------------------------------------------------------------

def temperature(h, delta_isa=0.0):
    """
    Température atmosphérique T(h, ΔISA).

    Paramètres
    ----------
    h : float ou array-like
        Altitude [m]
    delta_isa : float
        Déviation de température par rapport à l'ISA standard [°C]

    Retourne
    --------
    float ou ndarray
        Température [K]
    """
    h = np.asarray(h, dtype=float)

    # Troposphère : T décroît linéairement avec h (gradient L)
    T_tropo  = (T0 + delta_isa) + L * h

    # Stratosphère basse : T constante (gradient nul)
    T_strato = T11 + delta_isa

    # np.where permet de gérer les tableaux d'altitudes sans boucle
    return np.where(h <= H_TROPO, T_tropo, T_strato)


def pressure(h, delta_isa=0.0):
    """
    Pression atmosphérique P(h, ΔISA).

    Paramètres
    ----------
    h : float ou array-like / pour pouvoir facilement tracer des courbes plutard
        Altitude [m]
    delta_isa : float
        Déviation de température par rapport à l'ISA standard [°C]

    Retourne
    --------
    float ou ndarray
        Pression [Pa]
    """
    h = np.asarray(h, dtype=float)

    # Températures de référence avec déviation ISA appliquée
    T_sl     = T0 + delta_isa   # température au sol (base du calcul barométrique)
    T_strato = T11 + delta_isa  # température isotherme en stratosphère

    # --- Troposphère : formule barométrique à gradient constant ---
    # Découle de l'intégration de dP/P = -g/(R·T) dh avec T = T_sl + L·h
    T_tropo = T_sl + L * h
    P_tropo = P0 * (T_tropo / T_sl) ** _EXP

    # --- Pression à la tropopause (sert de référence pour la strato) ---
    P11_dev = P0 * (T_strato / T_sl) ** _EXP

    # --- Stratosphère ---
    # T = constante → P(h) = P11 · exp(−g·(h−11000) / (R·T_strato))
    P_strato = P11_dev * np.exp(-G * (h - H_TROPO) / (R * T_strato))

    return np.where(h <= H_TROPO, P_tropo, P_strato)


def density(h, delta_isa=0.0):
    """
    Masse volumique ρ(h, ΔISA) — loi des gaz parfaits : ρ = P / (R·T).

    Paramètres
    ----------
    h : float ou array-like
        Altitude [m]
    delta_isa : float
        Déviation de température par rapport à l'ISA standard [°C]

    Retourne
    --------
    float ou ndarray
        Masse volumique [kg/m³]
    """
    # Application directe de la loi des gaz parfaits : P = ρ·R·T
    return pressure(h, delta_isa) / (R * temperature(h, delta_isa))


def speed_of_sound(h, delta_isa=0.0):
    """
    Vitesse du son a(h, ΔISA) — gaz parfait : a = √(γ·R·T).

    Paramètres
    ----------
    h : float ou array-like
        Altitude [m]
    delta_isa : float
        Déviation de température par rapport à l'ISA standard [°C]

    Retourne
    --------
    float ou ndarray
        Vitesse du son [m/s]
    """
    # La vitesse du son ne dépend que de la température (pas de la pression)
    return np.sqrt(GAMMA * R * temperature(h, delta_isa))


# ---------------------------------------------------------------------------
# Ratios atmosphériques (θ, δ, σ) — normalisés par les valeurs de refrerence 
# ---------------------------------------------------------------------------

def theta(h, delta_isa=0.0):
    """Ratio de température θ = T / T0  [-]."""
    return temperature(h, delta_isa) / T0


def delta(h, delta_isa=0.0):
    """Ratio de pression δ = P / P0  [-]."""
    return pressure(h, delta_isa) / P0


def sigma(h, delta_isa=0.0):
    """Ratio de masse volumique σ = ρ / ρ0  [-]."""
    return density(h, delta_isa) / RHO0


# ---------------------------------------------------------------------------
# Inversion : pression → altitude
# ---------------------------------------------------------------------------

def altitude_from_pressure(P_query, delta_isa=0.0):
    """
    Calcule l'altitude correspondant à une pression donnée (inversion de P(h)).

    Troposphère  : P = P0·(T(h)/T_sl)^_EXP  →  h = T_sl·((P/P0)^(1/_EXP)−1) / L
    Stratosphère : P = P11·exp(−g·(h−11000)/(R·T_strato))  →  h = 11000 − R·T_strato/g·ln(P/P11)

    Paramètres
    ----------
    P_query : float
        Pression [Pa]  (doit vérifier 0 < P_query ≤ P0)
    delta_isa : float
        Déviation de température ISA [°C]

    Retourne
    --------
    float — Altitude [m]

    Lève
    ----
    ValueError si la pression est hors domaine.
    """
    # Vérification du domaine : P doit être positive et ≤ P au sol
    if P_query <= 0 or P_query > P0:
        raise ValueError(f"Pression hors domaine : P = {P_query:.2f} Pa (domaine : ]0, {P0:.0f}] Pa)")

    # Températures de référence avec déviation ISA
    T_sl     = T0 + delta_isa
    T_strato = T11 + delta_isa

    # Pression à la tropopause : sert de seuil pour choisir la couche
    P11_dev = P0 * (T_strato / T_sl) ** _EXP

    if P_query >= P11_dev:
        # --- Inversion troposphère : formule barométrique ---
        h = T_sl * ((P_query / P0) ** (1.0 / _EXP) - 1.0) / L
    else:
        # --- Inversion stratosphère : log de la loi isotherme ---
        h = H_TROPO - (R * T_strato / G) * np.log(P_query / P11_dev)

    return float(h)


# ---------------------------------------------------------------------------
# Fonction utilitaire — toutes les propriétés en une seule fois
# ---------------------------------------------------------------------------

def atmosphere(h, delta_isa=0.0):
    """
    Calcule l'ensemble des propriétés atmosphériques et des ratios.

    Paramètres
    ----------
    h : float ou array-like
        Altitude [m]
    delta_isa : float
        Déviation de température par rapport à l'ISA standard [°C]

    Retourne
    --------
    dict avec les clés :
        T      — Température [K]
        P      — Pression [Pa]
        rho    — Masse volumique [kg/m³]
        a      — Vitesse du son [m/s]
        theta  — Ratio de température T/T0   [-]
        delta  — Ratio de pression P/P0      [-]
        sigma  — Ratio de masse volumique ρ/ρ0 [-]
    """
    # Calcul de T et P une seule fois pour éviter les appels redondants
    T   = temperature(h, delta_isa)
    P   = pressure(h, delta_isa)

    # Masse volumique et vitesse du son déduites de T et P
    rho = P / (R * T)           # loi des gaz parfaits
    a   = np.sqrt(GAMMA * R * T)  # vitesse du son adiabatique

    return {
        "T":     T,
        "P":     P,
        "rho":   rho,
        "a":     a,
        "theta": T   / T0,    # ratio de température
        "delta": P   / P0,    # ratio de pression
        "sigma": rho / RHO0,  # ratio de masse volumique
    }
