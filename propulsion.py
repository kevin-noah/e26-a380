"""
Module de Propulsion — modèle moteur de l'Airbus A380

Modélise la poussée corrigée F*_th en fonction de la vitesse fan speed
corrigée N_fan_cor et du nombre de Mach, via un polynôme de surface
calibré sur données moteur.

Dépendances : numpy
"""

import numpy as np
import atmosphere as mod_atm

NOM         = "Module de Propulsion"
DESCRIPTION = "Poussée, débit carburant et émissions — A380"


# ---------------------------------------------------------------------------
# Polynôme de surface — poussée corrigée F*_th = f̄_th(N_fan_cor, M)
# ---------------------------------------------------------------------------
# Coefficients du fit polynomial (degré 5 en x=N_fan_cor, degré 3 en y=Mach)

_P = {
    'p00': +0.0072538779, 'p01': -0.0152486559, 'p02': -0.2872341189,
    'p03': +0.1855958598, 'p10': -0.0018632306, 'p11': -0.0015972827,
    'p12': +0.0047803814, 'p13': -0.0067222190, 'p20': +0.0002148330,
    'p21': -0.0000421767, 'p22': +0.0001528177, 'p23': +0.0000359784,
    'p30': -0.0000052656, 'p31': -0.0000014625, 'p32': -0.0000011862,
    'p40': +0.0000000713, 'p41': +0.0000000128, 'p50': -0.0000000003,
}


def f_fn_corrected(n_cor, mach):
    """
    Poussée corrigée F*_th via le polynôme de surface.

    Paramètres
    ----------
    n_cor : float ou array-like
        Vitesse fan speed corrigée N_fan_cor
    mach : float ou array-like
        Nombre de Mach

    Retourne
    --------
    float ou ndarray — poussée corrigée F*_th
    """
    x = np.asarray(n_cor, dtype=float)
    y = np.asarray(mach,  dtype=float)
    p = _P
    return (p['p00']
            + p['p10']*x          + p['p01']*y
            + p['p20']*x**2       + p['p11']*x*y        + p['p02']*y**2
            + p['p30']*x**3       + p['p21']*x**2*y     + p['p12']*x*y**2     + p['p03']*y**3
            + p['p40']*x**4       + p['p31']*x**3*y     + p['p22']*x**2*y**2  + p['p13']*x*y**3
            + p['p50']*x**5       + p['p41']*x**4*y     + p['p32']*x**3*y**2  + p['p23']*x**2*y**3)


# ---------------------------------------------------------------------------
# Polynôme de surface — débit carburant corrigé W*_f = f̄_wf(N_fan_cor, M)
# ---------------------------------------------------------------------------

_P_WF = {
    'p00': -0.001557817987, 'p01': +0.014020968264, 'p02': +0.006629228294,
    'p03': -0.026175772666, 'p10': +0.000372741439, 'p11': -0.005864395066,
    'p12': +0.006022804654, 'p13': +0.000021874793, 'p20': +0.000011163272,
    'p21': +0.000158667185, 'p22': -0.000165619582, 'p23': +0.000000911818,
    'p30': +0.000000084786, 'p31': -0.000001111082, 'p32': +0.000001161472,
    'p40': -0.000000012528, 'p41': -0.000000000126, 'p50': +0.000000000130,
}


def f_wf_corrected(n_cor, mach):
    """
    Débit carburant corrigé W*_f via le polynôme de surface (×2).

    Paramètres
    ----------
    n_cor : float ou array-like  Vitesse fan speed corrigée N_fan_cor
    mach  : float ou array-like  Nombre de Mach

    Retourne
    --------
    float ou ndarray — débit carburant corrigé W*_f
    """
    x = np.asarray(n_cor, dtype=float)
    y = np.asarray(mach,  dtype=float)
    p = _P_WF
    return (p['p00']
            + p['p10']*x          + p['p01']*y
            + p['p20']*x**2       + p['p11']*x*y        + p['p02']*y**2
            + p['p30']*x**3       + p['p21']*x**2*y     + p['p12']*x*y**2     + p['p03']*y**3
            + p['p40']*x**4       + p['p31']*x**3*y     + p['p22']*x**2*y**2  + p['p13']*x*y**3
            + p['p50']*x**5       + p['p41']*x**4*y     + p['p32']*x**3*y**2  + p['p23']*x**2*y**3
            ) * 2


# ---------------------------------------------------------------------------
# Poussée réelle — déscorrection par δ et √θ
# ---------------------------------------------------------------------------

def get_thrust(n1, mach, altitude, delta_isa=0.0):
    """
    Poussée nette réelle d'un moteur FN [N ou unité du polynôme].

    Correction : N1_cor = N1 / √θ   →   F*_th = f̄_th(N1_cor, M)   →   FN = F*_th × δ

    Paramètres
    ----------
    n1       : float  Vitesse fan speed réelle N1
    mach     : float  Nombre de Mach
    altitude : float  Altitude [m]
    delta_isa: float  Déviation de température ISA [°C]  (défaut : 0.0)

    Retourne
    --------
    float — poussée nette FN
    """
    theta = mod_atm.theta(altitude, delta_isa)
    delta = mod_atm.delta(altitude, delta_isa)
    n1_cor = n1 / np.sqrt(theta)
    f_star = f_fn_corrected(n1_cor, mach)
    return f_star * delta


def get_fuel_flow(n1, mach, altitude, delta_isa=0.0):
    """
    Débit carburant réel d'un moteur WF.

    Correction : N1_cor = N1 / √θ   →   W*_f = f̄_wf(N1_cor, M)   →   WF = W*_f × δ√θ

    Paramètres
    ----------
    n1       : float  Vitesse fan speed réelle N1
    mach     : float  Nombre de Mach
    altitude : float  Altitude [m]
    delta_isa: float  Déviation de température ISA [°C]  (défaut : 0.0)

    Retourne
    --------
    float — débit carburant WF
    """
    theta = mod_atm.theta(altitude, delta_isa)
    delta = mod_atm.delta(altitude, delta_isa)
    n1_cor = n1 / np.sqrt(theta)
    w_star = f_wf_corrected(n1_cor, mach)
    return w_star * delta * np.sqrt(theta)


# ---------------------------------------------------------------------------
# Émissions OACI — méthode Boeing Fuel Flow (BFF)
# ---------------------------------------------------------------------------
# Données de référence du moteur Trent 970B-84 (Banque de données OACI),
# aux quatre régimes du cycle LTO, classées par débit carburant croissant :
# IDLE (7 %), Approche (30 %), Climb (85 %), TO/GA (100 %).

_WF_C_REF   = np.array([0.261, 0.749, 2.262,  2.738])   # kg/s
_EI_NOX_REF = np.array([5.57, 12.37, 31.30,  41.34])    # g/kg
_EI_UHC_REF = np.array([0.04,  0.07,  0.09,   0.01])    # g/kg
_EI_CO_REF  = np.array([12.65, 1.05,  0.31,   0.33])    # g/kg

EI_CO2 = 3.16          # kg/kg — indice constant (modèle proportionnel)

_RH    = 0.80          # humidité relative de l'air
_X_BFF = 1.0           # exposants de la correction du débit (Ghazi et Botez)
_Y_BFF = 0.5


def _interp_ei_ref(wf_c_ref):
    """
    Indices d'émission corrigés de référence par interpolation linéaire
    sur diagramme ln-ln (EI vs débit carburant corrigé de référence).

    Paramètres
    ----------
    wf_c_ref : float — débit carburant corrigé de référence W*_F,C^REF [kg/s]

    Retourne
    --------
    dict — {'NOx', 'UHC', 'CO'} indices de référence [g/kg]
    """
    ln_wf = np.log(wf_c_ref)
    ln_wf_ref = np.log(_WF_C_REF)
    return {
        'NOx': np.exp(np.interp(ln_wf, ln_wf_ref, np.log(_EI_NOX_REF))),
        'UHC': np.exp(np.interp(ln_wf, ln_wf_ref, np.log(_EI_UHC_REF))),
        'CO':  np.exp(np.interp(ln_wf, ln_wf_ref, np.log(_EI_CO_REF))),
    }


def f_humidity(altitude, delta_isa=0.0):
    """
    Facteur spécifique d'humidité ω utilisé par la correction NOx.

    P_SAT = 6.107 × 10^(7.5·(T−273.15)/(T−35.85))   (T statique en kelvins)
    ω = 0.62197058 · RH · P_SAT / (0.1·P − RH·P_SAT)   avec RH = 0.80

    Paramètres
    ----------
    altitude : float  Altitude [m]
    delta_isa: float  Déviation de température ISA [°C]  (défaut : 0.0)

    Retourne
    --------
    float — facteur spécifique d'humidité ω
    """
    T = mod_atm.temperature(altitude, delta_isa)
    P = mod_atm.pressure(altitude, delta_isa)
    p_sat = 6.107 * 10.0 ** (7.5 * (T - 273.15) / (T - 35.85))
    return 0.62197058 * _RH * p_sat / (0.1 * P - _RH * p_sat)


def get_emission_indices(n1, mach, altitude, delta_isa=0.0):
    """
    Indices d'émission généralisés EI aux conditions de vol (méthode BFF).

    Étapes :
    1. WF réel  →  débit corrigé W_F,C = WF / (δ√θ)
    2. Débit de référence équivalent :
       W_F,C^REF = (1 + 0.2·M²)·θ^(3.8+y) / δ^(1−x) · W_F,C   (x=1, y=0.5)
    3. Interpolation ln-ln des indices de référence EI^REF
    4. Décorrélation vers les conditions de vol :
       EI_UHC = EI_UHC,C^REF · θ^3.3 / δ^1.02
       EI_CO  = EI_CO,C^REF  · θ^3.3 / δ^1.02
       EI_NOx = EI_NOx,C^REF · √(δ^1.02/θ^3.3) · exp[−19(ω − 0.00634)]

    Paramètres
    ----------
    n1       : float  Vitesse fan speed réelle N1
    mach     : float  Nombre de Mach
    altitude : float  Altitude [m]
    delta_isa: float  Déviation de température ISA [°C]  (défaut : 0.0)

    Retourne
    --------
    dict — {'EI_NOx', 'EI_UHC', 'EI_CO' [g/kg], 'EI_CO2' [kg/kg],
            'WF' [kg/s], 'WF_C_REF' [kg/s], 'omega'}
    """
    theta = mod_atm.theta(altitude, delta_isa)
    delta = mod_atm.delta(altitude, delta_isa)

    wf = get_fuel_flow(n1, mach, altitude, delta_isa)
    wf_c = wf / (delta * np.sqrt(theta))
    wf_c_ref = ((1.0 + 0.2 * mach**2) * theta**(3.8 + _Y_BFF)
                / delta**(1.0 - _X_BFF)) * wf_c

    ei_ref = _interp_ei_ref(wf_c_ref)
    omega = f_humidity(altitude, delta_isa)

    ei_uhc = ei_ref['UHC'] * theta**3.3 / delta**1.02
    ei_co  = ei_ref['CO']  * theta**3.3 / delta**1.02
    ei_nox = (ei_ref['NOx'] * np.sqrt(delta**1.02 / theta**3.3)
              * np.exp(-19.0 * (omega - 0.00634)))

    return {
        'EI_NOx':   ei_nox,
        'EI_UHC':   ei_uhc,
        'EI_CO':    ei_co,
        'EI_CO2':   EI_CO2,
        'WF':       wf,
        'WF_C_REF': wf_c_ref,
        'omega':    omega,
    }


def get_emissions(n1, mach, altitude, delta_isa=0.0, duration=1.0):
    """
    Masses de polluants émises par un moteur : Δm_i = EI_i × ΔF_B.

    Paramètres
    ----------
    n1       : float  Vitesse fan speed réelle N1
    mach     : float  Nombre de Mach
    altitude : float  Altitude [m]
    delta_isa: float  Déviation de température ISA [°C]  (défaut : 0.0)
    duration : float  Durée d'émission [s]  (défaut : 1.0 → taux par seconde)

    Retourne
    --------
    dict — {'m_NOx', 'm_UHC', 'm_CO' [g], 'm_CO2' [kg],
            'fuel_burn' [kg], + clés de get_emission_indices}
    """
    ei = get_emission_indices(n1, mach, altitude, delta_isa)
    fuel_burn = ei['WF'] * duration
    return {
        **ei,
        'fuel_burn': fuel_burn,
        'm_NOx':     ei['EI_NOx'] * fuel_burn,
        'm_UHC':     ei['EI_UHC'] * fuel_burn,
        'm_CO':      ei['EI_CO']  * fuel_burn,
        'm_CO2':     ei['EI_CO2'] * fuel_burn,
    }


# ---------------------------------------------------------------------------
# Visualisation 2D et 3D — FN et WF en fonction de N1 et Mach
# ---------------------------------------------------------------------------

def plot_prop(altitude, delta_isa=0.0,
              n1_range=(60.0, 100.0), mach_range=(0.0, 0.85), n_pts=40):
    """
    Affiche FN et WF en 2D (courbes par Mach) et 3D (surfaces) pour une altitude donnée.

    Paramètres
    ----------
    altitude  : float   Altitude [m]
    delta_isa : float   Déviation ISA [°C]  (défaut : 0.0)
    n1_range  : tuple   (N1_min, N1_max)    (défaut : 60–100)
    mach_range: tuple   (M_min, M_max)      (défaut : 0.0–0.85)
    n_pts     : int     Nombre de points par axe (défaut : 40)
    """
    import matplotlib.pyplot as plt

    n1s   = np.linspace(n1_range[0],   n1_range[1],   n_pts)
    machs = np.linspace(mach_range[0], mach_range[1], 6)

    fn_grid = np.array([[get_thrust(n, M, altitude, delta_isa)
                         for M in machs] for n in n1s])
    wf_grid = np.array([[get_fuel_flow(n, M, altitude, delta_isa)
                         for M in machs] for n in n1s])

    title_suffix = f"  h = {altitude:.0f} m  |  ΔISA = {delta_isa:+.1f} °C"

    # ── Figure 1 : courbes 2D ─────────────────────────────────────────────
    fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig1.suptitle(f'Propulsion — A380{title_suffix}', fontsize=13)

    def _plot2d(ax, data, ylabel):
        ax.plot(n1s, data, linewidth=0.9)
        ax.set_xlabel('N1')
        ax.set_ylabel(ylabel)
        ax.legend([f'M={M:.2f}' for M in machs], fontsize=7, loc='best')
        ax.grid(True)

    _plot2d(axes[0], fn_grid, 'FN')
    _plot2d(axes[1], wf_grid, 'WF')
    plt.tight_layout()

    # ── Figure 2 : surfaces 3D ────────────────────────────────────────────
    machs_3d  = np.linspace(mach_range[0], mach_range[1], n_pts)
    fn_surf   = np.array([[get_thrust(n, M, altitude, delta_isa)
                           for M in machs_3d] for n in n1s])
    wf_surf   = np.array([[get_fuel_flow(n, M, altitude, delta_isa)
                           for M in machs_3d] for n in n1s])

    fig2, axes3 = plt.subplots(1, 2, figsize=(14, 6),
                                subplot_kw={'projection': '3d'})
    fig2.suptitle(f'Surfaces propulsion — A380{title_suffix}', fontsize=13)

    N1_grid, M_grid = np.meshgrid(n1s, machs_3d)

    def _plot3d(ax, data, zlabel):
        ax.plot_surface(N1_grid, M_grid, data.T,
                        cmap='viridis', edgecolor='none', alpha=0.85)
        ax.view_init(elev=20, azim=45)
        ax.set_xlabel('N1')
        ax.set_ylabel('Mach')
        ax.set_zlabel(zlabel)

    _plot3d(axes3[0], fn_surf, 'FN')
    _plot3d(axes3[1], wf_surf, 'WF')
    plt.tight_layout()

    plt.show()


def plot_emissions(altitude, delta_isa=0.0,
                   n1_range=(60.0, 100.0), mach_range=(0.0, 0.85), n_pts=40):
    """
    Affiche les indices d'émission EI_NOx, EI_CO et EI_UHC en fonction
    de N1 (courbes par Mach) pour une altitude donnée.

    Paramètres
    ----------
    altitude  : float   Altitude [m]
    delta_isa : float   Déviation ISA [°C]  (défaut : 0.0)
    n1_range  : tuple   (N1_min, N1_max)    (défaut : 60–100)
    mach_range: tuple   (M_min, M_max)      (défaut : 0.0–0.85)
    n_pts     : int     Nombre de points par axe (défaut : 40)
    """
    import matplotlib.pyplot as plt

    n1s   = np.linspace(n1_range[0],   n1_range[1],   n_pts)
    machs = np.linspace(mach_range[0], mach_range[1], 6)

    grids = {'EI_NOx': [], 'EI_CO': [], 'EI_UHC': []}
    for n in n1s:
        rows = {k: [] for k in grids}
        for M in machs:
            ei = get_emission_indices(n, M, altitude, delta_isa)
            for k in grids:
                rows[k].append(ei[k])
        for k in grids:
            grids[k].append(rows[k])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Émissions OACI (BFF) — A380'
                 f'  h = {altitude:.0f} m  |  ΔISA = {delta_isa:+.1f} °C',
                 fontsize=13)

    for ax, (key, grid) in zip(axes, grids.items()):
        ax.plot(n1s, np.array(grid), linewidth=0.9)
        ax.set_xlabel('N1')
        ax.set_ylabel(f'{key} [g/kg]')
        ax.legend([f'M={M:.2f}' for M in machs], fontsize=7, loc='best')
        ax.grid(True)

    plt.tight_layout()
    plt.show()
