"""
Module de Conversion — conversions de vitesses aéronautiques.

Six fonctions de conversion entre TAS, CAS et Mach,
en fonction de l'altitude h [m] et de la déviation ΔISA [°C].

Formules utilisées (gaz parfait, écoulement isentropique subsonique) :

  TAS  ↔  Mach :
      V_TAS = M · a0 · √θ
      M     = V_TAS / (a0 · √θ)

  Mach → CAS :
      V_CAS = a0 · √( 5 · { [δ·{(1+0.2·M²)^3.5 − 1} + 1]^(1/3.5) − 1 } )

  CAS → Mach :
      M = √( 5 · { [1/δ · {[1+0.2·(V_CAS/a0)²]^3.5 − 1} + 1]^(1/3.5) − 1 } )

  TAS  ↔  CAS : passage obligatoire par le nombre de Mach.

Où :
  θ = T(h, ΔISA) / T0   — ratio de température (fourni par atmosphere.py)
  δ = P(h, ΔISA) / P0   — ratio de pression    (fourni par atmosphere.py)
  a0 = 340.294 m/s      — vitesse du son au niveau de la mer (ISA)
"""

import numpy as np
import atmosphere as _atm  # préfixe _ pour signaler usage interne uniquement

# Métadonnées du module (utilisées par main.py pour le menu)
NOM         = "Module de Conversion"
DESCRIPTION = "Conversions TAS / CAS / Mach en fonction de h et ΔISA"

# Vitesse du son MSL — constante de référence pour toutes les formules CAS
_A0 = _atm.A0    # 340.294 m/s


# ---------------------------------------------------------------------------
# 1. TAS ↔ Mach
#    Relation simple : TAS = M · a(h)  avec  a(h) = a0 · √θ
# ---------------------------------------------------------------------------

def mach_to_tas(M, h, delta_isa=0.0):
    """
    Calcule la TAS à partir du nombre de Mach.

        V_TAS = M · a0 · √θ

    Paramètres
    ----------
    M : float
        Nombre de Mach [-]
    h : float
        Altitude [m]
    delta_isa : float
        Déviation de température ISA [°C]

    Retourne
    --------
    float — True Airspeed [m/s]
    """
    # θ = T/T0 ; la vitesse du son locale vaut a = a0·√θ
    theta = _atm.theta(h, delta_isa)
    return M * _A0 * np.sqrt(theta)


def tas_to_mach(V_TAS, h, delta_isa=0.0):
    """
    Calcule le nombre de Mach à partir de la TAS.

        M = V_TAS / (a0 · √θ)

    Paramètres
    ----------
    V_TAS : float
        True Airspeed [m/s]
    h : float
        Altitude [m]
    delta_isa : float
        Déviation de température ISA [°C]

    Retourne
    --------
    float — Nombre de Mach [-]
    """
    # Division de la TAS par la vitesse du son locale a = a0·√θ
    theta = _atm.theta(h, delta_isa)
    return V_TAS / (_A0 * np.sqrt(theta))


# ---------------------------------------------------------------------------
# 2. CAS ↔ Mach
#    La CAS est définie par la pression différentielle pitot ramenée
#    aux conditions MSL (P0, a0) — d'où la dépendance en δ = P/P0.
# ---------------------------------------------------------------------------

def mach_to_cas(M, h, delta_isa=0.0):
    """
    Calcule la CAS à partir du nombre de Mach.

        V_CAS = a0 · √( 5 · { [δ·{(1+0.2·M²)^3.5 − 1} + 1]^(1/3.5) − 1 } )

    Paramètres
    ----------
    M : float
        Nombre de Mach [-]
    h : float
        Altitude [m]
    delta_isa : float
        Déviation de température ISA [°C]

    Retourne
    --------
    float — Calibrated Airspeed [m/s]
    """
    delta = _atm.delta(h, delta_isa)  # ratio de pression δ = P/P0

    # Pression différentielle pitot normalisée à l'altitude h
    # puis ramenée aux conditions sol pour obtenir la CAS
    inner = delta * ((1 + 0.2 * M**2)**3.5 - 1) + 1
    return _A0 * np.sqrt(5 * (inner**(1/3.5) - 1))


def cas_to_mach(V_CAS, h, delta_isa=0.0):
    """
    Calcule le nombre de Mach à partir de la CAS.

        M = √( 5 · { [1/δ · {[1+0.2·(V_CAS/a0)²]^3.5 − 1} + 1]^(1/3.5) − 1 } )

    Paramètres
    ----------
    V_CAS : float
        Calibrated Airspeed [m/s]
    h : float
        Altitude [m]
    delta_isa : float
        Déviation de température ISA [°C]

    Retourne
    --------
    float — Nombre de Mach [-]
    """
    delta = _atm.delta(h, delta_isa)  # ratio de pression δ = P/P0

    # Pression différentielle depuis la CAS (conditions sol), puis
    # divisée par δ pour retrouver les conditions à l'altitude h
    inner = (1 / delta) * ((1 + 0.2 * (V_CAS / _A0)**2)**3.5 - 1) + 1
    return np.sqrt(5 * (inner**(1/3.5) - 1))


# ---------------------------------------------------------------------------
# 3. TAS ↔ CAS  (passage obligatoire par le Mach)
# ---------------------------------------------------------------------------

def tas_to_cas(V_TAS, h, delta_isa=0.0):
    """
    Convertit la TAS en CAS.

        V_TAS → M = tas_to_mach(V_TAS) → V_CAS = mach_to_cas(M)

    Paramètres
    ----------
    V_TAS : float
        True Airspeed [m/s]
    h : float
        Altitude [m]
    delta_isa : float
        Déviation de température ISA [°C]

    Retourne
    --------
    float — Calibrated Airspeed [m/s]
    """
    # Étape 1 : TAS → Mach via la vitesse du son locale
    M = tas_to_mach(V_TAS, h, delta_isa)
    # Étape 2 : Mach → CAS via la pression locale et P0
    return mach_to_cas(M, h, delta_isa)


def cas_to_tas(V_CAS, h, delta_isa=0.0):
    """
    Convertit la CAS en TAS.

        V_CAS → M = cas_to_mach(V_CAS) → V_TAS = mach_to_tas(M)

    Paramètres
    ----------
    V_CAS : float
        Calibrated Airspeed [m/s]
    h : float
        Altitude [m]
    delta_isa : float
        Déviation de température ISA [°C]

    Retourne
    --------
    float — True Airspeed [m/s]
    """
    # Étape 1 : CAS → Mach via la pression locale et P0
    M = cas_to_mach(V_CAS, h, delta_isa)
    # Étape 2 : Mach → TAS via la vitesse du son locale
    return mach_to_tas(M, h, delta_isa)
