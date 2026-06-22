"""
Module d'Équilibrage (Trim) — équilibrage longitudinal de l'Airbus A380

Résout, pour une configuration avion (masse, centrage) et une condition de vol
(Mach, altitude, ΔISA), les trois inconnues de l'équilibre longitudinal en
palier :

    α       — angle d'incidence            [deg]
    δstab   — calage du stabilisateur (THS) [deg]
    F_N     — poussée totale (4 moteurs)    [N]

Système d'équations en palier (slides MGA803) :

    0 = F_N·sin(α+φ_T) + L − m·g₀                 (force verticale)
    0 = F_N·cos(α+φ_T) − D                        (force horizontale)
    0 = M_aero(α, δstab) + M_moteur(F_N)          (moment de tangage)

avec  L = q̄·S_wb·CL_s(α,δstab),  D = q̄·S_wb·CD_s(α,δstab),  q̄ = ½·ρ·V_T².

Méthode de résolution : algorithme itératif de point fixe proposé par
Ghazi & Botez (voir slides). Une fois l'avion équilibré, on inverse le modèle
de poussée pour remonter au régime N1 puis au débit carburant W_F.

Dépendances : numpy, scipy (atmosphere, aerodynamics, propulsion).
"""

import numpy as np
from scipy.optimize import brentq

import atmosphere   as mod_atm
import aerodynamics as mod_aero
import propulsion   as mod_prop

NOM         = "Module d'Équilibrage (Trim)"
DESCRIPTION = "Équilibrage longitudinal en palier — α, δstab, F_N, W_F (A380)"

# ---------------------------------------------------------------------------
# Constantes physiques et géométriques
# ---------------------------------------------------------------------------

G0    = mod_atm.G          # accélération gravitationnelle [m/s²]
S_WB  = mod_aero._S_WB     # surface de référence aile+fuselage [m²]  (859.0)
C_WB  = mod_aero._C_WB     # corde aérodynamique moyenne (MAC)  [m]   (11.0)

N_ENGINES = 4              # nombre de moteurs (A380 : 4)

# ───────────────────────────────────────────────────────────────────────────
#  DONNÉES PROPULSIVES / CENTRAGE — À CONFIRMER AVEC LES SLIDES DU COURS
# ───────────────────────────────────────────────────────────────────────────
# Tant que les positions moteurs restent nulles, la ligne de poussée passe par
# le foyer (M_moteur = 0) et δstab équilibre le seul moment aéro + poids.
# Renseigner ces valeurs (point de référence : foyer CA à 25 % MAC) active la
# contribution moteur au moment de tangage.

PHI_T = 2.0        # angle d'inclinaison de la poussée φ_T              [deg]
C_W   = C_WB       # corde aéro moyenne de l'aile c̄_w (= c̄_wb ?)      [m]   ⚠

# Positions des moteurs (données cours, en mètres)
X_ENG_23 = 6.30   # position longitudinale paire intérieure (moteurs 2-3) [m]
Z_ENG_23 = 1.6   # position verticale     paire intérieure (moteurs 2-3) [m]
X_ENG_14 = 1.7   # position longitudinale paire extérieure (moteurs 1-4) [m]
Z_ENG_14 = 0.5   # position verticale     paire extérieure (moteurs 1-4) [m]
# ───────────────────────────────────────────────────────────────────────────

# Régime moteur de référence (inversion de poussée et estimé initial F_N^max)
N1_MAX = 100.0     # N1 maximal [%]
N1_MIN = 20.0      # N1 minimal pour l'inversion de poussée [%]

# Facteur d'échelle de la poussée : Newtons par unité du polynôme propulsion.
# get_thrust() est normalisé par la poussée maximale F_N^max = 352.90 kN
# (sortie ≈ 1.0 au régime max) ; l'équilibre travaille en N (poids mg₀, q·S).
FN_MAX = 352_900.0      # poussée maximale d'UN moteur F_N^max [N]  (352.90 kN)
THRUST_SCALE = FN_MAX   # [N / (unité get_thrust)]  — get_thrust normalisé par moteur

# Le débit carburant get_fuel_flow() est lui aussi normalisé, par le débit
# maximal d'un moteur W_F^max ; le débit physique = get_fuel_flow × W_F^max.
WF_MAX_KGH = 9_856.8               # débit carburant maximal d'UN moteur W_F^max [kg/h]
FUEL_SCALE = WF_MAX_KGH / 3600.0   # [kg/s par unité get_fuel_flow]  (= 2.738 kg/s)


def _thrust_N(n1, mach, altitude, delta_isa=0.0):
    """Poussée d'un moteur exprimée en Newtons (get_thrust × THRUST_SCALE)."""
    return mod_prop.get_thrust(n1, mach, altitude, delta_isa) * THRUST_SCALE


def _fuel_flow_kgs(n1, mach, altitude, delta_isa=0.0):
    """Débit carburant d'un moteur en kg/s (get_fuel_flow × FUEL_SCALE)."""
    return mod_prop.get_fuel_flow(n1, mach, altitude, delta_isa) * FUEL_SCALE

# Bornes de recherche des inconnues (cohérentes avec les plages du modèle aéro)
DSTAB_MIN, DSTAB_MAX = -15.0, 15.0   # plage de calage du stabilisateur [deg]

# Critères de convergence (Ghazi & Botez)
EPS_ALPHA = 1.0e-3   # tolérance sur α      [deg]
EPS_FN    = 1.0e+1   # tolérance sur F_N    [N]
EPS_DSTAB = 1.0e-3   # tolérance sur δstab  [deg]
MAX_ITER  = 100      # nombre maximal d'itérations


# ---------------------------------------------------------------------------
# Moments de tangage
# ---------------------------------------------------------------------------

def m_moteur(fn_total, alpha):
    """
    Moment de tangage des moteurs autour du foyer CA.

    M_moteur =  x23·F23·sin(α+φ_T) + z23·F23·cos(α+φ_T)
              + z14·F14·cos(α+φ_T) − x14·F14·sin(α+φ_T)

    Hypothèse symétrique : 4 moteurs identiques → chaque paire porte la moitié
    de la poussée totale (F23 = F14 = F_N/2).

    Paramètres
    ----------
    fn_total : float  Poussée totale des 4 moteurs [N]
    alpha    : float  Angle d'incidence [deg]

    Retourne
    --------
    float — moment moteur [N·m]
    """
    ang = np.radians(alpha + PHI_T)
    s, c = np.sin(ang), np.cos(ang)
    f23 = fn_total / 2.0
    f14 = fn_total / 2.0
    return (X_ENG_23 * f23 * s + Z_ENG_23 * f23 * c
            + Z_ENG_14 * f14 * c - X_ENG_14 * f14 * s)


def m_aero(model, alpha, mach, dstab, q, weight, gamma=0.0):
    """
    Moment de tangage aérodynamique + poids autour du foyer CA.

    M_aero = q̄·S_wb·c̄_wb·C_ms(α,δstab) − (-x_cg + 0.25)·c̄_w·W·cos(γ)

    Le terme de poids est inclus dans `weight` via l'appelant ; ici on ne traite
    que la part aérodynamique (le terme de centrage est ajouté dans
    `moment_total`, qui connaît x_cg).
    """
    cm = mod_aero.get_cm_total(model, alpha, mach, delta_it=dstab)
    return q * S_WB * C_WB * cm


def moment_total(model, alpha, mach, dstab, fn_total, q, weight, x_cg, gamma=0.0):
    """
    Moment de tangage total autour du foyer CA (nul à l'équilibre).

    M_tot = q̄·S_wb·c̄_wb·C_ms − (0.25 − x_cg)·c̄_w·W·cos(γ) + M_moteur(F_N)
    """
    m_a = m_aero(model, alpha, mach, dstab, q, weight, gamma)
    m_poids = (0.25 - x_cg) * C_W * weight * np.cos(np.radians(gamma))
    m_m = m_moteur(fn_total, alpha)
    return m_a - m_poids + m_m


# ---------------------------------------------------------------------------
# Inversions 1D (recherche de racine par brentq)
# ---------------------------------------------------------------------------

def _solve_alpha(model, cl_target, mach, dstab):
    """Cherche α* tel que CL_total(α*, M, δstab) = cl_target (sur la plage du modèle)."""
    a_min = float(model['f_clwb']['x_alpha'][0])
    a_max = float(model['f_clwb']['x_alpha'][-1])
    f = lambda a: mod_aero.get_cl_total(model, a, mach, delta_it=dstab) - cl_target
    if f(a_min) * f(a_max) > 0:
        raise ValueError(
            f"CL requis = {cl_target:.4f} hors de la plage atteignable "
            f"α∈[{a_min:.1f}°, {a_max:.1f}°] (δstab={dstab:.2f}°).")
    return brentq(f, a_min, a_max, xtol=1e-6)


def _solve_dstab(model, alpha, mach, fn_total, q, weight, x_cg, gamma):
    """Cherche δstab* qui annule le moment de tangage total."""
    f = lambda d: moment_total(model, alpha, mach, d, fn_total, q, weight, x_cg, gamma)
    if f(DSTAB_MIN) * f(DSTAB_MAX) > 0:
        raise ValueError(
            f"Aucun δstab dans [{DSTAB_MIN:.0f}°, {DSTAB_MAX:.0f}°] n'annule "
            f"le moment (α={alpha:.2f}°, M={mach:.3f}).")
    return brentq(f, DSTAB_MIN, DSTAB_MAX, xtol=1e-6)


def n1_from_thrust(fn_engine, mach, altitude, delta_isa=0.0):
    """
    Inverse le modèle de poussée : trouve le régime N1 [%] qui produit la
    poussée `fn_engine` (par moteur) au Mach et à l'altitude donnés.
    """
    f = lambda n: _thrust_N(n, mach, altitude, delta_isa) - fn_engine/4
    fmin, fmax = f(N1_MIN), f(N1_MAX)
    if fmin * fmax > 0:
        fn_max = _thrust_N(N1_MAX, mach, altitude, delta_isa)
        raise ValueError(
            f"Poussée requise {fn_engine:.0f} N/moteur hors d'atteinte "
            f"(max ≈ {fn_max:.0f} N à N1={N1_MAX:.0f} %).")
    return brentq(f, N1_MIN, N1_MAX, xtol=1e-4)


# ---------------------------------------------------------------------------
# Algorithme d'équilibrage (Ghazi & Botez)
# ---------------------------------------------------------------------------

def trim(mass, mach, altitude, delta_isa=0.0, x_cg=0.40, gamma=0.0,
         model=None, verbose=False,
         eps_alpha=EPS_ALPHA, eps_fn=EPS_FN, eps_dstab=EPS_DSTAB):
    """
    Équilibre l'avion en palier et retourne les paramètres de trim + le débit
    carburant.

    Paramètres
    ----------
    mass      : float  Masse de l'avion [kg]
    mach      : float  Nombre de Mach
    altitude  : float  Altitude [m]
    delta_isa : float  Déviation de température ISA [°C]   (défaut : 0)
    x_cg      : float  Centrage — position du CG en fraction de MAC
                       (bras du poids = (0.25−x_cg)·c̄_w)   (défaut : 0.40 = 40 %)
    gamma     : float  Pente de la trajectoire [deg]       (défaut : 0 = palier)
    model     : dict   Modèle aéro (build_aero_model). Construit si None.
    verbose   : bool   Affiche l'historique des itérations.
    eps_alpha : float  Tolérance de convergence sur α      [deg] (défaut EPS_ALPHA)
    eps_fn    : float  Tolérance de convergence sur F_N     [N]   (défaut EPS_FN)
    eps_dstab : float  Tolérance de convergence sur δstab   [deg] (défaut EPS_DSTAB)

    Retourne
    --------
    dict — {'alpha', 'dstab', 'FN', 'FN_engine', 'N1', 'WF_engine', 'WF_total',
            'CL', 'CD', 'L', 'D', 'finesse', 'iterations', 'converged', ...}
    """
    if model is None:
        model = mod_aero.build_aero_model()

    # Conditions de vol
    rho = float(mod_atm.density(altitude, delta_isa))
    a   = float(mod_atm.speed_of_sound(altitude, delta_isa))
    V   = mach * a
    q   = 0.5 * rho * V**2
    weight = mass * G0

    # Estimé initial (it. 0) : α = 0°, δstab = 0°, F_N⁰ = 40 % de F_N^max
    # F_N^max est la poussée max statique des 4 moteurs (4 × 352.9 kN ≈ 1411 kN),
    # pas la poussée disponible déclassée en altitude → F_N⁰ ≈ 565 kN.
    fn_max_total = N_ENGINES * FN_MAX
    alpha = 0.0
    dstab = 0.0
    fn    = 0.40 * fn_max_total

    converged = False
    # Itération 0 : point de départ de l'algorithme (avant toute résolution).
    history = [{
        'it':      0,
        'alpha':   alpha,
        'dstab':   dstab,
        'FN':      fn,
        'CL':      None,
        'CD':      None,
        'd_alpha': None,
        'd_FN':    None,
        'd_dstab': None,
    }]

    for it in range(1, MAX_ITER + 1):
        # 1. Portance nécessaire pour équilibrer le poids (palier)
        L = weight - fn * np.sin(np.radians(alpha + PHI_T))
        cl_s = L / (q * S_WB)

        # 2. Incidence α* qui fournit ce CL (à δstab courant)
        alpha_star = _solve_alpha(model, cl_s, mach, dstab)

        # 3. Traînée à α* puis poussée nécessaire pour l'équilibrer
        cd_s = mod_aero.get_cd_total(model, alpha_star, mach, delta_it=dstab)
        D = q * S_WB * cd_s
        fn_star = D / np.cos(np.radians(alpha_star + PHI_T))

        # 4. Calage δstab* qui annule le moment total (α*, F_N*)
        dstab_star = _solve_dstab(model, alpha_star, mach, fn_star,
                                  q, weight, x_cg, gamma)

        # 5. Test de convergence
        d_alpha = abs(alpha - alpha_star)
        d_fn    = abs(fn - fn_star)
        d_dstab = abs(dstab - dstab_star)
        history.append({
            'it':      it,
            'alpha':   alpha_star,
            'dstab':   dstab_star,
            'FN':      fn_star,
            'CL':      cl_s,
            'CD':      cd_s,
            'd_alpha': d_alpha,
            'd_FN':    d_fn,
            'd_dstab': d_dstab,
        })
        if verbose:
            print(f"  it {it:2d}  α={alpha_star:7.3f}°  δstab={dstab_star:7.3f}°  "
                  f"F_N={fn_star:11.1f} N  |Δα|={d_alpha:.2e} "
                  f"|ΔF_N|={d_fn:.2e} |Δδ|={d_dstab:.2e}")

        alpha, dstab, fn = alpha_star, dstab_star, fn_star

        if d_alpha <= eps_alpha and d_fn <= eps_fn and d_dstab <= eps_dstab:
            converged = True
            break

    # Grandeurs finales à l'équilibre
    cl_s = mod_aero.get_cl_total(model, alpha, mach, delta_it=dstab)
    cd_s = mod_aero.get_cd_total(model, alpha, mach, delta_it=dstab)
    L = q * S_WB * cl_s
    D = q * S_WB * cd_s
    finesse = cl_s / cd_s if cd_s != 0 else float('inf')

    # Inversion de la poussée → N1 → débit carburant.
    # Si la poussée requise dépasse la poussée max disponible (avion limité en
    # poussée à ce point de vol), l'équilibre aéro reste valide (α, δstab, F_N,
    # historique) mais N1/W_F ne sont pas définissables → on les laisse à None
    # et on lève un drapeau, sans planter, pour que l'app/CLI affichent quand
    # même le résultat et le tableau d'itérations.
    fn_engine = fn / N_ENGINES
    thrust_limited = False
    n1 = wf_engine = wf_total = wf_total_kgh = None
    try:
        n1 = n1_from_thrust(fn_engine, mach, altitude, delta_isa)
        wf_engine = _fuel_flow_kgs(n1, mach, altitude, delta_isa)   # [kg/s]
        wf_total  = N_ENGINES * wf_engine                            # [kg/s]
        wf_total_kgh = wf_total * 3600.0                             # [kg/h]
    except ValueError:
        thrust_limited = True

    return {
        'alpha':      alpha,
        'dstab':      dstab,
        'FN':         fn,
        'FN_engine':  fn_engine,
        'N1':         n1,
        'WF_engine':  wf_engine,           # [kg/s]  (None si limité en poussée)
        'WF_total':   wf_total,            # [kg/s]  (None si limité en poussée)
        'WF_total_kgh': wf_total_kgh,      # [kg/h]  (None si limité en poussée)
        'CL':         cl_s,
        'CD':         cd_s,
        'L':          L,
        'D':          D,
        'finesse':    finesse,
        'q':          q,
        'V':          V,
        'weight':     weight,
        'iterations': it,
        'converged':  converged,
        'thrust_limited': thrust_limited,
        'eps_alpha':  eps_alpha,
        'eps_fn':     eps_fn,
        'eps_dstab':  eps_dstab,
        'history':    history,
    }
