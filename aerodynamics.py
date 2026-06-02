"""
Module Aérodynamique — modèle aérodynamique de l'Airbus A380

Charge les fichiers .history générés par VSPAERO (OpenVSP) et construit
des tables d'interpolation 2D : CL, CD, Cm en fonction de α [deg] et Mach.

Structure des données (aero_data) :
    f_clwb, f_cdwb, f_cmwb  — aile + fuselage  (Wing-Body)
    f_clht, f_cdht           — empennage arrière (Horizontal Tail)
    Chaque entrée : dict avec x_alpha [deg], y_mach [-], value [n_α × n_M]

Dépendances : numpy, pandas, scipy  (matplotlib pour la visualisation)
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline

NOM         = "Module Aérodynamique"
DESCRIPTION = "CL, CD, Cm en fonction de α et Mach — A380 (OpenVSP)"

# Chemin par défaut vers les fichiers de données (dossier data/ du projet)
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DEFAULT_FILE_WB = os.path.join(_DATA_DIR, 'Avion_WB_VSPGeom.history')
DEFAULT_FILE_HT = os.path.join(_DATA_DIR, 'Avion_HT_VSPGeom.history')
DEFAULT_FILE_GEOM = os.path.join(_DATA_DIR, 'Avion_WBT_VSPGeom.vspgeom')

# Colonnes extraites du fichier .history OpenVSP (Iter col.0 et T/QS col.19 exclus)
_COLS = [
    'Mach', 'AoA', 'Beta', 'CL', 'CDo', 'CDi', 'CDtot', 'CDt',
    'CDtot_t', 'CS', 'LoD', 'E', 'CFx', 'CFy', 'CFz', 'CMx', 'CMy', 'CMz',
]


# ---------------------------------------------------------------------------
# Lecture du fichier .history OpenVSP
# ---------------------------------------------------------------------------

def load_history(filename):
    """
    Lit un fichier .history VSPAERO et retourne un DataFrame pandas.

    Pour chaque Solver Case, extrait la dernière itération convergée
    (5e itération = 7e ligne après la balise "Solver Case:").

    Paramètres
    ----------
    filename : str
        Chemin vers le fichier .history

    Retourne
    --------
    pd.DataFrame  (colonnes : Mach, AoA, Beta, CL, CDo, …, CMz)

    Lève
    ----
    FileNotFoundError si le fichier est introuvable.
    ValueError si aucun Solver Case valide n'est trouvé.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Fichier introuvable : {filename}")

    rows = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if 'Solver Case:' in line:
            target = i + 7
            if target < len(lines):
                vals = np.fromstring(lines[target], dtype=float, sep=' ')
                if len(vals) == 20:
                    rows.append(vals[1:19])   # supprimer Iter (0) et T/QS (19)

    if not rows:
        raise ValueError(f"Aucun Solver Case valide trouvé dans : {filename}")

    return pd.DataFrame(rows, columns=_COLS)


# ---------------------------------------------------------------------------
# Construction du modèle d'interpolation
# ---------------------------------------------------------------------------

def _build_grid(df, col):
    """
    Construit une grille d'interpolation 2D pour la colonne `col`.

    Retourne un dict avec :
        x_alpha  — vecteur des angles d'attaque [deg]
        y_mach   — vecteur des nombres de Mach
        value    — matrice (n_alpha × n_mach)
        _interp  — RectBivariateSpline pour l'interpolation bicubique
    """
    alphas = np.unique(df['AoA'].values)
    machs  = np.unique(df['Mach'].values)
    grid   = np.zeros((len(alphas), len(machs)))

    for j, mach in enumerate(machs):
        mask = df['Mach'] == mach
        grid[:, j] = df.loc[mask, col].values

    return {
        'x_alpha': alphas,
        'y_mach':  machs,
        'value':   grid,
        '_interp': RectBivariateSpline(alphas, machs, grid),
    }


def build_aero_model(file_wb=DEFAULT_FILE_WB, file_ht=DEFAULT_FILE_HT):
    """
    Charge les deux fichiers .history et construit le modèle aérodynamique.

    Paramètres
    ----------
    file_wb : str
        Chemin vers le fichier .history aile + fuselage. Par défaut : data/Avion_WB_VSPGeom.history
    file_ht : str
        Chemin vers le fichier .history empennage arrière. Par défaut : data/Avion_HT_VSPGeom.history

    Retourne
    --------
    dict avec les clés : f_clwb, f_cdwb, f_cmwb, f_clht, f_cdht
        Chaque valeur est un dict : x_alpha, y_mach, value, _interp
    """
    df_wb = load_history(file_wb)
    df_ht = load_history(file_ht)

    return {
        'f_clwb': _build_grid(df_wb, 'CL'),
        'f_cdwb': _build_grid(df_wb, 'CDtot'),
        'f_cmwb': _build_grid(df_wb, 'CMy'),
        'f_clht': _build_grid(df_ht, 'CL'),
        'f_cdht': _build_grid(df_ht, 'CDtot'),
        'f_cmht': _build_grid(df_ht, 'CMy'),
    }


# ---------------------------------------------------------------------------
# Fonctions d'interpolation publiques
# ---------------------------------------------------------------------------

def _interp(grid, alpha, mach):
    return float(grid['_interp']([alpha], [mach], grid=False)[0])


def get_cl_wb(model, alpha, mach):
    """CL de l'aile + fuselage pour alpha [deg] et Mach."""
    return _interp(model['f_clwb'], alpha, mach)


def get_cd_wb(model, alpha, mach):
    """CD total de l'aile + fuselage pour alpha [deg] et Mach."""
    return _interp(model['f_cdwb'], alpha, mach)


def get_cm_wb(model, alpha, mach):
    """Cm (moment de tangage) de l'aile + fuselage pour alpha [deg] et Mach."""
    return _interp(model['f_cmwb'], alpha, mach)


# ---------------------------------------------------------------------------
# Downwash — angle de déviation de l'écoulement de l'aile vers l'empennage
# ---------------------------------------------------------------------------

# Coefficients du modèle linéaire : ε = ε0 + εα × α [deg]
_EPS0      = 1.18   # downwash à incidence nulle [deg]
_EPS_ALPHA = 0.37   # gradient de downwash       [deg/deg]


def f_downwash(alpha, eps0=_EPS0, eps_alpha=_EPS_ALPHA):
    """
    Calcule l'angle de downwash ε à l'empennage arrière.

    Modèle linéaire :  ε = ε0 + εα × α

    Paramètres
    ----------
    alpha : float ou array-like
        Angle d'attaque de l'aile [deg]
    eps0 : float
        Downwash à incidence nulle [deg]  (défaut : 1.18)
    eps_alpha : float
        Gradient de downwash [deg/deg]   (défaut : 0.37)

    Retourne
    --------
    float ou ndarray — angle de downwash ε [deg]
    """
    alpha = np.asarray(alpha, dtype=float)
    return eps0 + eps_alpha * alpha


def get_cl_ht(model, alpha, mach, delta_it=0.0):
    """CL de l'empennage arrière pour alpha [deg] et Mach."""
    alpha_ht = alpha - f_downwash(alpha) + delta_it
    return _interp(model['f_clht'], alpha_ht, mach)


def get_cd_ht(model, alpha, mach, delta_it=0.0):
    """CD total de l'empennage arrière pour alpha [deg] et Mach."""
    alpha_ht = alpha - f_downwash(alpha) + delta_it
    return _interp(model['f_cdht'], alpha_ht, mach)


def get_cm_ht(model, alpha, mach, delta_it=0.0):
    """Cm (moment de tangage) de l'empennage arrière pour alpha [deg] et Mach."""
    alpha_ht = alpha - f_downwash(alpha) + delta_it
    return _interp(model['f_cmht'], alpha_ht, mach)


# ---------------------------------------------------------------------------
# Géométrie de référence A380 (vraies dimensions)
# ---------------------------------------------------------------------------

_S_WB = 859.0   # surface de référence aile + fuselage   [m²]
_C_WB = 11.0    # corde aérodynamique moyenne aile (MAC)  [m]
_S_HT = 205.0   # surface de référence stabilisateur HT   [m²]
_C_HT = 6.77    # corde aérodynamique moyenne empennage   [m]
_X_HT = 32.0    # bras de levier longitudinal HT          [m]
_Z_HT = 1.24    # bras de levier vertical HT              [m]


# ---------------------------------------------------------------------------
# Coefficients totaux avion (Wing-Body + empennage HT)
# ---------------------------------------------------------------------------

def get_cl_total(model, alpha, mach, delta_it=0.0):
    """
    CL total de l'avion.

    CL_t = CL_wb + (S_ht/S_wb) * (CL_ht*cos(ε) - CD_ht*sin(ε))
    """
    eps_rad = np.radians(float(f_downwash(alpha)))
    cl_wb = _interp(model['f_clwb'], alpha, mach)
    cl_ht = get_cl_ht(model, alpha, mach, delta_it=delta_it)
    cd_ht = get_cd_ht(model, alpha, mach, delta_it=delta_it)
    return cl_wb + (_S_HT / _S_WB) * (cl_ht * np.cos(eps_rad) - cd_ht * np.sin(eps_rad))


def get_cd_total(model, alpha, mach, delta_it=0.0):
    """
    CD total de l'avion.

    CD_t = CD_wb + (S_ht/S_wb) * (CD_ht*cos(ε) + CL_ht*sin(ε))
    """
    eps_rad = np.radians(float(f_downwash(alpha)))
    cd_wb = _interp(model['f_cdwb'], alpha, mach)
    cl_ht = get_cl_ht(model, alpha, mach, delta_it=delta_it)
    cd_ht = get_cd_ht(model, alpha, mach, delta_it=delta_it)
    return cd_wb + (_S_HT / _S_WB) * (cd_ht * np.cos(eps_rad) + cl_ht * np.sin(eps_rad))


def get_cm_total(model, alpha, mach, delta_it=0.0):
    """
    CM total de l'avion.

    CM_t = CM_wb + (S_ht*c_ht / S_wb*c_wb) * CM_ht
           - (S_ht*x̄_ht / S_wb*c_wb) * (CL_ht*cos(ε) - CD_ht*sin(ε))
           + (S_ht*z̄_ht / S_wb*c_wb) * (CL_ht*cos(ε) + CD_ht*sin(ε))
    avec x̄_ht = x_ht*cos(ε) - z_ht*sin(ε)
         z̄_ht = z_ht*cos(ε) - x_ht*sin(ε)
    """
    eps_rad = np.radians(float(f_downwash(alpha)))
    cos_e   = np.cos(eps_rad)
    sin_e   = np.sin(eps_rad)

    x_bar = _X_HT * cos_e - _Z_HT * sin_e
    z_bar = _Z_HT * cos_e - _X_HT * sin_e

    cm_wb = _interp(model['f_cmwb'], alpha, mach)
    cm_ht = get_cm_ht(model, alpha, mach, delta_it=delta_it)
    cl_ht = get_cl_ht(model, alpha, mach, delta_it=delta_it)
    cd_ht = get_cd_ht(model, alpha, mach, delta_it=delta_it)

    ratio = _S_HT / (_S_WB * _C_WB)
    return (cm_wb
            + (_S_HT * _C_HT / (_S_WB * _C_WB)) * cm_ht
            - ratio * x_bar * (cl_ht * cos_e - cd_ht * sin_e)
            + ratio * z_bar * (cl_ht * cos_e + cd_ht * sin_e))


# ---------------------------------------------------------------------------
# Visualisation 2D et 3D des coefficients aérodynamiques
# ---------------------------------------------------------------------------

def plot_aero_model(model):
    """
    Affiche les courbes 2D et surfaces 3D des coefficients aérodynamiques.

    Figure 1 : CL_wb, CD_wb, CL_ht, CD_ht vs alpha (une courbe par Mach).
    Figure 2 : mêmes données en surfaces 3D f(alpha, Mach).

    Paramètres
    ----------
    model : dict  retourné par build_aero_model()
    """
    import matplotlib.pyplot as plt

    # ── Figure 1 : courbes 2D ─────────────────────────────────────────────
    fig1, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig1.suptitle('Coefficients aérodynamiques — A380', fontsize=13)

    def _plot2d(ax, grid, ylabel):
        ax.plot(grid['x_alpha'], grid['value'], linewidth=0.9)
        ax.set_xlabel(r'$\alpha$ [deg]')
        ax.set_ylabel(ylabel)
        ax.grid(True)

    _plot2d(axes[0, 0], model['f_clwb'], r'$C_{L_{wb}}$')
    _plot2d(axes[0, 1], model['f_cdwb'], r'$C_{D_{wb}}$')
    _plot2d(axes[1, 0], model['f_clht'], r'$C_{L_{ht}}$')
    _plot2d(axes[1, 1], model['f_cdht'], r'$C_{D_{ht}}$')
    plt.tight_layout()

    # ── Figure 2 : surfaces 3D ────────────────────────────────────────────
    fig2, axes3 = plt.subplots(2, 2, figsize=(13, 9),
                                subplot_kw={'projection': '3d'})
    fig2.suptitle('Surfaces aérodynamiques — A380', fontsize=13)

    def _plot3d(ax, grid, zlabel):
        X, Y = np.meshgrid(grid['x_alpha'], grid['y_mach'])
        ax.plot_surface(X, Y, grid['value'].T,
                        cmap='viridis', edgecolor='none', alpha=0.85)
        ax.view_init(elev=20, azim=45)
        ax.set_xlabel(r'$\alpha$ [deg]')
        ax.set_ylabel('Mach')
        ax.set_zlabel(zlabel)

    _plot3d(axes3[0, 0], model['f_clwb'], r'$C_{L_{wb}}$')
    _plot3d(axes3[0, 1], model['f_cdwb'], r'$C_{D_{wb}}$')
    _plot3d(axes3[1, 0], model['f_clht'], r'$C_{L_{ht}}$')
    _plot3d(axes3[1, 1], model['f_cdht'], r'$C_{D_{ht}}$')
    plt.tight_layout()

    plt.show()


def plot_total(model, delta_it=0.0):
    """
    Affiche les courbes 2D et surfaces 3D des coefficients totaux CL_t, CD_t, CM_t.

    Figure 1 : CL_t, CD_t, CM_t vs alpha (une courbe par Mach).
    Figure 2 : mêmes données en surfaces 3D f(alpha, Mach).

    Paramètres
    ----------
    model    : dict  retourné par build_aero_model()
    delta_it : float  calage de l'empennage [deg] (défaut : 0.0)
    """
    import matplotlib.pyplot as plt

    ref    = model['f_clwb']
    alphas = ref['x_alpha']
    machs  = ref['y_mach']

    # Calcul des grilles totales  [n_alpha × n_mach]
    cl_t = np.array([[get_cl_total(model, a, M, delta_it=delta_it)
                      for M in machs] for a in alphas])
    cd_t = np.array([[get_cd_total(model, a, M, delta_it=delta_it)
                      for M in machs] for a in alphas])
    cm_t = np.array([[get_cm_total(model, a, M, delta_it=delta_it)
                      for M in machs] for a in alphas])

    title_suffix = f"  (δit = {delta_it:.1f}°)" if delta_it != 0.0 else ""

    # ── Figure 1 : courbes 2D ─────────────────────────────────────────────
    fig1, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig1.suptitle(f'Coefficients totaux — A380{title_suffix}', fontsize=13)

    def _plot2d(ax, data, ylabel):
        ax.plot(alphas, data, linewidth=0.9)
        ax.set_xlabel(r'$\alpha$ [deg]')
        ax.set_ylabel(ylabel)
        ax.legend([f'M={M:.3f}' for M in machs], fontsize=7, loc='best')
        ax.grid(True)

    _plot2d(axes[0], cl_t, r'$C_{L_t}$')
    _plot2d(axes[1], cd_t, r'$C_{D_t}$')
    _plot2d(axes[2], cm_t, r'$C_{M_t}$')
    plt.tight_layout()

    # ── Figure 2 : surfaces 3D ────────────────────────────────────────────
    fig2, axes3 = plt.subplots(1, 3, figsize=(16, 6),
                                subplot_kw={'projection': '3d'})
    fig2.suptitle(f'Surfaces totales — A380{title_suffix}', fontsize=13)

    A, M_grid = np.meshgrid(alphas, machs)

    def _plot3d(ax, data, zlabel):
        ax.plot_surface(A, M_grid, data.T,
                        cmap='viridis', edgecolor='none', alpha=0.85)
        ax.view_init(elev=20, azim=45)
        ax.set_xlabel(r'$\alpha$ [deg]')
        ax.set_ylabel('Mach')
        ax.set_zlabel(zlabel)

    _plot3d(axes3[0], cl_t, r'$C_{L_t}$')
    _plot3d(axes3[1], cd_t, r'$C_{D_t}$')
    _plot3d(axes3[2], cm_t, r'$C_{M_t}$')
    plt.tight_layout()

    plt.show()


# ---------------------------------------------------------------------------
# Visualisation 3D de la géométrie OpenVSP (.vspgeom)
# ---------------------------------------------------------------------------

def show_geometry(fname=DEFAULT_FILE_GEOM):
    """
    Affiche la géométrie 3D de l'avion depuis un fichier .vspgeom OpenVSP.

    Format du fichier (texte, tokens séparés par des espaces) :
        npt              — nombre de points du maillage
        3 × npt floats   — coordonnées X, Y, Z (ordre colonne, comme MATLAB fscanf)
        npoly            — nombre de triangles
        4 × npoly ints   — connectivité (col 0 = nb sommets, cols 1–3 = indices 1-based)
        7 × npoly floats — données surface (col 0 = ID surface)

    Paramètres
    ----------
    fname : str
        Chemin vers le fichier .vspgeom. Par défaut : data/Avion_WBT_VSPGeom.vspgeom

    Lève
    ----
    FileNotFoundError si le fichier est introuvable.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if not os.path.isfile(fname):
        raise FileNotFoundError(f"Fichier introuvable : {fname}")

    with open(fname, 'r') as f:
        tokens = f.read().split()

    idx = 0

    npt    = int(tokens[idx]); idx += 1
    # order='F' reproduit la lecture colonne par colonne de MATLAB fscanf([3 npt])
    ptdata = np.array(tokens[idx:idx + 3 * npt], dtype=float).reshape((3, npt), order='F')
    idx   += 3 * npt

    npoly   = int(tokens[idx]); idx += 1
    condata = np.array(tokens[idx:idx + 4 * npoly], dtype=int).reshape((4, npoly), order='F')
    idx    += 4 * npoly
    con = condata[1:4, :] - 1   # cols 1–3, passage à l'indexation 0-based Python

    # Lecture des données surface pour valider le parsing (non affiché ici)
    _ = np.array(tokens[idx:idx + 7 * npoly], dtype=float).reshape((7, npoly), order='F')

    fig = plt.figure(figsize=(11, 7))
    ax  = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(
        ptdata[0, :], ptdata[1, :], ptdata[2, :],
        triangles=con.T,
        color=(0.8, 0.8, 0.8),
        edgecolor='k',
        linewidth=0.05,
        alpha=0.9,
    )

    # Équivalent de 'axis equal' MATLAB pour les axes 3D matplotlib
    limits     = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers    = limits.mean(axis=1)
    half_range = (limits[:, 1] - limits[:, 0]).max() / 2
    ax.set_xlim3d(centers[0] - half_range, centers[0] + half_range)
    ax.set_ylim3d(centers[1] - half_range, centers[1] + half_range)
    ax.set_zlim3d(centers[2] - half_range, centers[2] + half_range)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Géométrie Airbus A380 — OpenVSP')
    ax.grid(True)
    plt.tight_layout()
    plt.show()
