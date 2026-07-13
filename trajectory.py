"""
Module de Prédiction de Trajectoire — profil vertical de croisière (A380)

Enchaîne des segments de croisière à altitude et Mach CONSTANTS, reliés par des
step-climbs INSTANTANÉS (saut d'altitude, masse et instant conservés). À chaque
pas de distance, l'avion est équilibré (trim.trim) → débit carburant W_F et
vitesse vraie TAS ; on intègre le temps de vol, le carburant brûlé (la masse
décroît au fil du vol) et les masses de polluants (indices d'émission OACI ×
carburant brûlé).

Ce module est le support des deux derniers livrables de l'énoncé :
  • point 2 — prédiction de trajectoire pour la phase de croisière (step-climbs) ;
  • point 4 — analyse de performance (temps, carburant, émissions) sur une
    trajectoire réelle, comparée sans / avec 1 / avec 2 step-climbs.

Hypothèses (énoncé) :
  • Seul le profil VERTICAL est modélisé (pas de route latérale, pas de montée
    initiale ni de descente : on ne traite QUE la phase de croisière).
  • Step-climbs INSTANTANÉS (note ² de l'énoncé) : l'altitude passe d'un palier
    au suivant sans phase de montée (masse et instant inchangés au saut).
  • VENT NUL : distance sol = distance air (temps et SR référencés à la TAS).
  • Masse et centrage : la masse décroît par combustion ; x_cg reste fixé
    (config énoncé 500 000 kg / CG 40 %).

Intégration : marche avant (Euler) par pas de distance `ds` fixe (défaut 25 NM,
la granularité de la méthode du cours). À chaque pas, l'avion est trimmé à la
masse courante ; le carburant brûlé sur le pas (≈ 0.1 % de la masse à 25 NM) est
petit devant la masse → l'erreur d'Euler est négligeable.

Dépendances : numpy (atmosphere, aerodynamics, trim, propulsion).
"""

import numpy as np

import atmosphere   as mod_atm
import aerodynamics as mod_aero
import propulsion   as mod_prop
import trim         as mod_trim

NOM         = "Module de Prédiction de Trajectoire"
DESCRIPTION = "Profil vertical de croisière + step-climbs — temps, carburant, émissions (A380)"

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

KT = 0.514444          # 1 nœud en m/s
NM = 1852.0            # 1 mille nautique en m
FT = 0.3048            # 1 pied en m

SUBSTEP_NM     = 25.0             # pas d'intégration standard [NM] (méthode du cours)
SUBSTEP_DEFAUT = SUBSTEP_NM * NM  # pas d'intégration en distance [m] (25 NM ≈ 46.3 km)
STEP_CLIMB_FT  = 2000.0           # amplitude d'un step-climb [ft] (séparation RVSM)

# Polluants suivis (clés d'indice d'émission → clé de masse cumulée).
_POLLUANTS = ('NOx', 'UHC', 'CO', 'CO2')


def _emissions_nulles():
    """Dict de masses de polluants cumulées, initialisé à zéro [kg]."""
    return {p: 0.0 for p in _POLLUANTS}


# ---------------------------------------------------------------------------
# Segment de croisière à altitude et Mach constants
# ---------------------------------------------------------------------------

def integrate_segment(mass0, distance, altitude, mach, delta_isa=0.0,
                      x_cg=0.40, model=None, ds=SUBSTEP_DEFAUT, s0=0.0, t0=0.0):
    """
    Intègre un segment de croisière à altitude et Mach CONSTANTS sur `distance`.

    À chaque pas de distance, l'avion est équilibré (trim) à la masse courante ;
    on en tire la TAS et le débit carburant total W_F, d'où le temps du pas
    (dt = ds/TAS), le carburant brûlé (dm = W_F·dt) et les masses de polluants
    (EI_i × dm, indices d'émission OACI au régime N1 d'équilibre).

    Paramètres
    ----------
    mass0     : float  Masse en début de segment [kg]
    distance  : float  Longueur du segment (distance parcourue) [m]
    altitude  : float  Altitude du palier [m]
    mach      : float  Nombre de Mach de croisière (constant sur le segment)
    delta_isa : float  Déviation ISA [°C]                     (défaut 0)
    x_cg      : float  Centrage (fraction de MAC)             (défaut 0.40)
    model     : dict   Modèle aéro (build_aero_model) ; construit si None
    ds        : float  Pas d'intégration en distance [m]      (défaut 25 NM ≈ 46.3 km)
    s0, t0    : float  Distance / temps cumulés en entrée de segment (pour
                       chaîner plusieurs segments dans un profil global)

    Retourne
    --------
    dict — {'mass_end', 'time', 'fuel', 'distance', 'emissions' {kg},
            'samples': [ {s, t, alt, mach, mass, tas, wf_kgh, N1, finesse}, … ]}

    Lève
    ----
    RuntimeError — si un point du segment est limité en poussée (W_F indéfini) :
        le palier est infaisable à cette masse (typiquement altitude trop haute
        pour une masse encore lourde → justifie le recours aux step-climbs).
    """
    if model is None:
        model = mod_aero.build_aero_model()

    n = max(1, int(np.ceil(distance / ds)))
    step = distance / n
    a = float(mod_atm.speed_of_sound(altitude, delta_isa))
    tas = mach * a

    mass = float(mass0)
    time = float(t0)
    s = float(s0)
    fuel = 0.0
    emis = _emissions_nulles()
    samples = []

    for _ in range(n):
        # Deux causes d'infaisabilité d'un palier, unifiées en RuntimeError :
        #   • trim aéro impossible (CL requis hors plage) → trim lève ValueError ;
        #   • limité en poussée (W_F/N1 indéfinis) → trim renvoie WF/N1 = None.
        try:
            res = mod_trim.trim(mass, mach, altitude, delta_isa=delta_isa,
                                x_cg=x_cg, model=model)
        except ValueError as exc:
            raise RuntimeError(
                f"Palier infaisable (équilibre aéro impossible) à "
                f"{altitude/FT:.0f} ft, masse {mass/1000:.1f} t, M{mach:.3f} : "
                f"{exc}") from exc
        wf = res['WF_total']                       # [kg/s]  (None si limité)
        if not wf or res['N1'] is None:
            raise RuntimeError(
                f"Palier infaisable (limité en poussée) à {altitude/FT:.0f} ft, "
                f"masse {mass/1000:.1f} t, M{mach:.3f} : W_F indéfini.")

        dt = step / tas                            # [s]
        dm = wf * dt                               # [kg] carburant brûlé sur le pas

        # Indices d'émission OACI au régime d'équilibre, puis masses (EI × dm).
        ei = mod_prop.get_emission_indices(res['N1'], mach, altitude, delta_isa)
        emis['NOx'] += ei['EI_NOx'] * dm / 1000.0  # g/kg × kg → g → kg
        emis['UHC'] += ei['EI_UHC'] * dm / 1000.0
        emis['CO']  += ei['EI_CO']  * dm / 1000.0
        emis['CO2'] += ei['EI_CO2'] * dm           # kg/kg × kg → kg

        samples.append({
            's':       s, 't': time, 'alt': altitude, 'mach': mach,
            'mass':    mass, 'tas': tas, 'wf_kgh': wf * 3600.0,
            'N1':      res['N1'], 'finesse': res['finesse'],
        })

        mass -= dm
        fuel += dm
        time += dt
        s    += step

    return {
        'mass_end':  mass,
        'time':      time - t0,
        'fuel':      fuel,
        'distance':  distance,
        'emissions': emis,
        'samples':   samples,
    }


# ---------------------------------------------------------------------------
# Enchaînement de segments (profil vertical complet avec step-climbs)
# ---------------------------------------------------------------------------

def fly(mass0, segments, delta_isa=0.0, x_cg=0.40, model=None,
        ds=SUBSTEP_DEFAUT):
    """
    Vole un profil vertical = liste de segments de croisière, reliés par des
    step-climbs instantanés.

    Paramètres
    ----------
    mass0    : float  Masse au début de la croisière [kg]
    segments : list de (distance [m], altitude [m], mach) — un palier chacun.
               Le passage d'un segment au suivant est un step-climb instantané
               (l'altitude change, la masse et l'instant sont conservés).
    delta_isa, x_cg, model, ds : cf. integrate_segment.

    Retourne
    --------
    dict — {'mass0', 'mass_end', 'time', 'fuel', 'distance',
            'n_step_climbs', 'emissions' {kg},
            'segments': [résultat de chaque integrate_segment],
            'profile': [échantillons cumulés, step-climbs marqués par un saut
                        d'altitude à distance constante]}
    """
    if model is None:
        model = mod_aero.build_aero_model()

    mass = float(mass0)
    time = 0.0
    s = 0.0
    fuel = 0.0
    emis = _emissions_nulles()
    seg_results = []
    profile = []

    for i, (distance, altitude, mach) in enumerate(segments):
        seg = integrate_segment(mass, distance, altitude, mach,
                                delta_isa=delta_isa, x_cg=x_cg, model=model,
                                ds=ds, s0=s, t0=time)
        seg_results.append(seg)
        profile.extend(seg['samples'])
        # Marqueur de fin de palier (altitude courante à la distance atteinte) :
        # rend le saut vertical visible sur le profil altitude(distance).
        mass = seg['mass_end']
        fuel += seg['fuel']
        time += seg['time']
        s    += seg['distance']
        for p in _POLLUANTS:
            emis[p] += seg['emissions'][p]
        profile.append({'s': s, 't': time, 'alt': altitude, 'mach': mach,
                        'mass': mass, 'tas': np.nan, 'wf_kgh': np.nan,
                        'N1': np.nan, 'finesse': np.nan})

    return {
        'mass0':         float(mass0),
        'mass_end':      mass,
        'time':          time,
        'fuel':          fuel,
        'distance':      s,
        'n_step_climbs': len(segments) - 1,
        'emissions':     emis,
        'segments':      seg_results,
        'profile':       profile,
    }


# ---------------------------------------------------------------------------
# Plans de step-climbs et comparaison 0 / 1 / 2 montées
# ---------------------------------------------------------------------------

def plan_egal(total_distance, levels, mach):
    """
    Répartit `total_distance` en parts ÉGALES entre les paliers `levels`.

    levels : liste d'altitudes [m] (un palier par segment). Renvoie la liste de
    segments (distance, altitude, mach) attendue par `fly`. Avec n paliers →
    (n-1) step-climbs, chacun à i/n de la distance totale.
    """
    n = len(levels)
    d = total_distance / n
    return [(d, alt, mach) for alt in levels]


def compare_step_climbs(mass0, total_distance, mach, base_level,
                        step_ft=STEP_CLIMB_FT, delta_isa=0.0, x_cg=0.40,
                        model=None, ds=SUBSTEP_DEFAUT):
    """
    Compare la MÊME trajectoire (même distance, même masse initiale, même Mach)
    volée avec 0, 1 puis 2 step-climbs, à partir du palier `base_level`.

      • 0 step-climb : tout le vol à base_level ;
      • 1 step-climb : moitié à base_level, moitié à base_level + step ;
      • 2 step-climbs : tiers à base_level, +step, +2·step.

    Un palier de step-climb standard (RVSM) = 2000 ft. Les cas dont un palier est
    infaisable (limité en poussée à la masse courante) sont renvoyés avec
    'feasible' = False et le message d'erreur, sans planter la comparaison.

    Paramètres
    ----------
    mass0          : float  Masse au début de croisière [kg]
    total_distance : float  Distance de croisière [m]
    mach           : float  Mach de croisière (constant)
    base_level     : float  Altitude du 1ᵉʳ palier [m]
    step_ft        : float  Amplitude d'un step-climb [ft]  (défaut 2000)
    delta_isa, x_cg, model, ds : cf. fly.

    Retourne
    --------
    dict — {0: cas, 1: cas, 2: cas} où chaque `cas` est
        {'n_step_climbs', 'levels' [m], 'feasible', 'error',
         'time', 'fuel', 'distance', 'emissions', 'result'(dict fly) }.
    """
    if model is None:
        model = mod_aero.build_aero_model()

    step = step_ft * FT
    cas = {}
    for k in (0, 1, 2):
        levels = [base_level + j * step for j in range(k + 1)]
        segments = plan_egal(total_distance, levels, mach)
        entry = {'n_step_climbs': k, 'levels': levels,
                 'feasible': True, 'error': None,
                 'time': np.nan, 'fuel': np.nan, 'distance': total_distance,
                 'emissions': _emissions_nulles(), 'result': None}
        try:
            res = fly(mass0, segments, delta_isa=delta_isa, x_cg=x_cg,
                      model=model, ds=ds)
            entry.update({'time': res['time'], 'fuel': res['fuel'],
                          'distance': res['distance'],
                          'emissions': res['emissions'], 'result': res})
        except RuntimeError as exc:
            entry['feasible'] = False
            entry['error'] = str(exc)
        cas[k] = entry
    return cas
