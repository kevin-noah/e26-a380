"""
Module de Performance de croisière — vitesses optimales MRC / LRC / ECON (A380)

À partir du module d'équilibrage (trim.trim), détermine, pour une masse et une
altitude données, les trois vitesses de croisière optimales :

    MRC  — Maximum Range Cruise  : le Mach qui MAXIMISE la portée spécifique
           SR = TAS / W_F  [mètres parcourus par kg de carburant brûlé].
           C'est le point où d(SR)/dM = 0.

    LRC  — Long Range Cruise     : le Mach, PLUS RAPIDE que MRC, où la portée
           spécifique vaut SR = 0.99 · SR_max (convention industrielle : on
           accepte ~1 % d'autonomie en moins pour gagner en vitesse).

    ECON — Economy Cruise        : le Mach qui MINIMISE le coût total
           « carburant + temps ». Avec un Cost Index CI (coût du temps exprimé
           en kg de carburant par minute), on minimise le coût par unité de
           distance :  C(M) = (W_F + CI/60) / TAS.
           Cas limites : CI = 0 → ECON ≡ MRC ; CI grand → ECON tend vers Mmo.

Deux techniques combinées (le sujet en laisse le choix) :
    • échantillonnage : balayage du Mach → courbes SR(M) et coût(M) (traçables) ;
    • optimisation    : raffinage de chaque optimum par scipy
                        (minimize_scalar borné pour MRC/ECON, brentq pour LRC).

Dépendances : numpy, scipy (trim, atmosphere).
"""

import numpy as np
from scipy.optimize import minimize_scalar, brentq

import atmosphere as mod_atm
import aerodynamics as mod_aero
import trim as mod_trim

NOM         = "Module de Performance de croisière"
DESCRIPTION = "Vitesses de croisière optimales — MRC, LRC, ECON (A380)"

# Plage de balayage par défaut (cohérente avec l'enveloppe aéro du modèle).
MACH_MIN_DEFAUT = 0.50
MACH_MAX_DEFAUT = 0.90
N_PTS_DEFAUT    = 41        # nombre de points d'échantillonnage du Mach
CI_DEFAUT       = 160.0     # Cost Index par défaut [kg/min] (point de réf. du devoir)

KT = 0.514444              # 1 nœud en m/s


# ---------------------------------------------------------------------------
# Portée spécifique et coût (grandeurs ponctuelles)
# ---------------------------------------------------------------------------

def _safe_trim(mass, mach, altitude, delta_isa, model, **trim_kw):
    """Appelle trim.trim en absorbant l'échec du trim aéro (CL hors plage).

    Renvoie le dict de trim, ou ``None`` si l'équilibre aérodynamique n'existe
    pas à ce point (ValueError). Un point simplement *limité en poussée* n'est
    PAS un échec : trim renvoie alors un dict avec W_F = None.
    """
    try:
        return mod_trim.trim(mass, mach, altitude, delta_isa=delta_isa,
                            model=model, **trim_kw)
    except ValueError:
        return None


def specific_range(mass, mach, altitude, delta_isa=0.0, model=None, **trim_kw):
    """
    Portée spécifique SR = TAS / W_F  [m parcourus par kg de carburant].

    Équilibre l'avion au point (masse, Mach, altitude) via trim.trim, puis
    rapporte la vitesse vraie au débit carburant total. Renvoie ``np.nan`` si
    le point est limité en poussée ou si l'équilibre aéro n'existe pas.
    """
    if model is None:
        model = mod_aero.build_aero_model()
    res = _safe_trim(mass, mach, altitude, delta_isa, model, **trim_kw)
    wf = res['WF_total'] if res else None       # [kg/s] (None si limité/échec)
    if not wf:                                  # None ou 0 → SR indéfinie
        return np.nan
    a   = float(mod_atm.speed_of_sound(altitude, delta_isa))
    tas = mach * a
    return tas / wf                            # [m / kg]


def _point_info(mass, mach, altitude, delta_isa, model, **trim_kw):
    """Détail d'un point de croisière (Mach, TAS, SR, W_F, finesse…)."""
    res = _safe_trim(mass, mach, altitude, delta_isa, model, **trim_kw)
    a   = float(mod_atm.speed_of_sound(altitude, delta_isa))
    tas = mach * a
    if res is None:                             # équilibre aéro inexistant
        return {'mach': mach, 'tas': tas, 'tas_kt': tas / KT, 'wf': None,
                'wf_kgh': None, 'sr': np.nan, 'sr_nm_per_kg': np.nan,
                'finesse': np.nan, 'alpha': np.nan, 'N1': None,
                'thrust_limited': False, 'failed': True}
    wf  = res['WF_total']                       # [kg/s]
    return {
        'mach':           mach,
        'tas':            tas,                   # [m/s]
        'tas_kt':         tas / KT,              # [kt]
        'wf':             wf,                    # [kg/s]
        'wf_kgh':         res['WF_total_kgh'],   # [kg/h]
        'sr':             (tas / wf) if wf else np.nan,   # [m/kg]
        'sr_nm_per_kg':   ((tas / wf) / 1852.0) if wf else np.nan,  # [NM/kg]
        'finesse':        res['finesse'],
        'alpha':          res['alpha'],
        'N1':             res['N1'],
        'thrust_limited': res.get('thrust_limited', False),
    }


def _cost_per_distance(sr, wf, tas, cost_index):
    """
    Coût par unité de distance  C = (W_F + CI/60) / TAS  [kg de carburant / m].

    CI (Cost Index) = coût du temps exprimé en kg de carburant par MINUTE.
    Renvoie ``np.nan`` si W_F/TAS indéfinis.
    """
    if not wf or not tas:
        return np.nan
    ci_kgs = cost_index / 60.0                  # kg/min → kg/s
    return (wf + ci_kgs) / tas


# ---------------------------------------------------------------------------
# Vitesses de croisière optimales — MRC / LRC / ECON
# ---------------------------------------------------------------------------

def cruise_speeds(mass, altitude, delta_isa=0.0, cost_index=CI_DEFAUT,
                  mach_min=MACH_MIN_DEFAUT, mach_max=MACH_MAX_DEFAUT,
                  n_pts=N_PTS_DEFAUT, model=None, refine=True, **trim_kw):
    """
    Détermine MRC, LRC et ECON à masse et altitude fixées.

    Étape 1 — échantillonnage : on balaie le Mach dans [mach_min, mach_max] et on
    calcule, à chaque pas, la portée spécifique SR(M) et le coût C(M) (via trim).
    Étape 2 — optimisation : on raffine chaque optimum (scipy) à partir du
    meilleur point échantillonné.

    Paramètres
    ----------
    mass       : float  Masse de l'avion [kg]
    altitude   : float  Altitude [m]
    delta_isa  : float  Déviation ISA [°C]                       (défaut 0)
    cost_index : float  Cost Index pour ECON [kg/min]            (défaut CI_DEFAUT=160)
    mach_min/max, n_pts : bornes et finesse du balayage
    model      : dict   Modèle aéro (build_aero_model) ; construit si None
    refine     : bool   Raffinage par optimiseur après l'échantillonnage

    Retourne
    --------
    dict — {'mass', 'altitude', 'delta_isa', 'cost_index',
            'curve': {'mach', 'tas', 'wf', 'sr', 'cost'},
            'MRC' : {...}, 'LRC' : {...}, 'ECON' : {...}}
           Chaque optimum est un dict détaillé (cf. _point_info) ; None si
           l'optimum n'a pu être déterminé (aucun point exploitable).
    """
    if model is None:
        model = mod_aero.build_aero_model()

    # --- Étape 1 : échantillonnage --------------------------------------
    machs = np.linspace(mach_min, mach_max, n_pts)
    a     = float(mod_atm.speed_of_sound(altitude, delta_isa))
    tas   = machs * a
    sr    = np.full(n_pts, np.nan)
    wf    = np.full(n_pts, np.nan)
    ld    = np.full(n_pts, np.nan)        # finesse L/D (dispo même si limité poussée)
    for i, M in enumerate(machs):
        res = _safe_trim(mass, float(M), altitude, delta_isa, model, **trim_kw)
        if res:
            ld[i] = res.get('finesse', np.nan)
            w = res['WF_total']
            if w:
                wf[i] = w
                sr[i] = tas[i] / w
    cost = np.array([_cost_per_distance(sr[i], wf[i], tas[i], cost_index)
                     for i in range(n_pts)])

    curve = {'mach': machs, 'tas': tas, 'wf': wf, 'sr': sr, 'cost': cost,
             'finesse': ld}

    if not np.any(np.isfinite(sr)):
        # Aucun point exploitable (tout limité en poussée).
        return {'mass': mass, 'altitude': altitude, 'delta_isa': delta_isa,
                'cost_index': cost_index, 'curve': curve,
                'MRC': None, 'LRC': None, 'ECON': None}

    # Fonctions scalaires bornées (np.nan → pénalité, écartées par l'optimiseur)
    def neg_sr(M):
        v = specific_range(mass, float(M), altitude, delta_isa, model, **trim_kw)
        return np.inf if not np.isfinite(v) else -v

    def cost_of(M):
        res = _safe_trim(mass, float(M), altitude, delta_isa, model, **trim_kw)
        w = res['WF_total'] if res else None
        v = _cost_per_distance((float(M) * a) / w if w else np.nan, w,
                               float(M) * a, cost_index)
        return np.inf if not np.isfinite(v) else v

    finite = np.isfinite(sr)

    # --- MRC : maximum de SR --------------------------------------------
    i_mrc = int(np.nanargmax(sr))
    m_mrc = machs[i_mrc]
    if refine:
        lo = machs[max(i_mrc - 1, 0)]
        hi = machs[min(i_mrc + 1, n_pts - 1)]
        if hi > lo:
            opt = minimize_scalar(neg_sr, bounds=(lo, hi), method='bounded')
            if opt.success and np.isfinite(opt.fun):
                m_mrc = float(opt.x)
    sr_max = specific_range(mass, m_mrc, altitude, delta_isa, model, **trim_kw)
    mrc = _point_info(mass, m_mrc, altitude, delta_isa, model, **trim_kw)
    mrc['label'] = 'MRC'

    # --- LRC : SR = 0.99 · SR_max, côté rapide (M > MRC) ----------------
    lrc = None
    target = 0.99 * sr_max
    # On cherche le croisement SR(M) = target pour M > m_mrc.
    mach_hi = [machs[i] for i in range(n_pts) if finite[i] and machs[i] > m_mrc]
    if mach_hi:
        def g(M):
            v = specific_range(mass, float(M), altitude, delta_isa, model, **trim_kw)
            return (v - target) if np.isfinite(v) else -1.0
        m_lo, m_hi = m_mrc, max(mach_hi)
        if g(m_lo) > 0 and g(m_hi) < 0:
            m_lrc = float(brentq(g, m_lo, m_hi, xtol=1e-4))
        elif g(m_hi) >= 0:
            m_lrc = m_hi          # SR reste ≥ target jusqu'au bord → LRC = bord
        else:
            m_lrc = m_mrc         # SR chute trop vite → LRC ≈ MRC
        lrc = _point_info(mass, m_lrc, altitude, delta_isa, model, **trim_kw)
        lrc['label'] = 'LRC'

    # --- ECON : minimum du coût (carburant + temps) ---------------------
    econ = None
    if np.any(np.isfinite(cost)):
        i_eco = int(np.nanargmin(cost))
        m_eco = machs[i_eco]
        if refine:
            lo = machs[max(i_eco - 1, 0)]
            hi = machs[min(i_eco + 1, n_pts - 1)]
            if hi > lo:
                opt = minimize_scalar(cost_of, bounds=(lo, hi), method='bounded')
                if opt.success and np.isfinite(opt.fun):
                    m_eco = float(opt.x)
        econ = _point_info(mass, m_eco, altitude, delta_isa, model, **trim_kw)
        econ['label'] = 'ECON'

    return {
        'mass':       mass,
        'altitude':   altitude,
        'delta_isa':  delta_isa,
        'cost_index': cost_index,
        'sr_max':     sr_max,
        'curve':      curve,
        'MRC':        mrc,
        'LRC':        lrc,
        'ECON':       econ,
    }
