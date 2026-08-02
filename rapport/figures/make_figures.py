"""
Génération des figures du rapport IEEE (MGA803 — A380).

Toutes les figures sont calculées par les modules RÉELS du projet (aucune
donnée rejouée) et exportées en PDF vectoriel aux largeurs IEEE :
colonne 3.5 in, pleine page 7.16 in.

Exécution (depuis la racine du dépôt) :
    .venv/bin/python rapport/figures/make_figures.py

Sorties (rapport/figures/) :
    fig_aero_surfaces.pdf   surfaces 3D CL/CD/CM (WB et HT)
    fig_ei_lnln.pdf         diagramme ln-ln des indices d'émission OACI
    fig_sr_optima.pdf       SR(M) et optima MRC/LRC/ECON (point de validation)
    fig_cout_ci.pdf         décomposition du coût carburant/temps/total
    fig_sr_masses.pdf       SR(M) pour plusieurs masses + lieux des optima
    fig_ek215.pdf           étude EK215 : profil, masse, carburant, comparaison
    valeurs.json            tous les chiffres cités dans le texte du rapport
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
OUT = os.path.dirname(os.path.abspath(__file__))

import atmosphere as mod_atm            # noqa: E402
import aerodynamics as mod_aero         # noqa: E402
import propulsion as mod_prop           # noqa: E402
import trim as mod_trim                 # noqa: E402
import performance as mod_perf          # noqa: E402
import trajectory as mod_traj           # noqa: E402

# ---------------------------------------------------------------------------
# Charte graphique (palette validée — validateur dataviz, surface blanche)
# ---------------------------------------------------------------------------

INK    = "#22272E"   # encre (courbe de référence, texte)
BLUE   = "#2563EB"   # série 1 (MRC · cas direct)
AMBER  = "#D97706"   # série 2 (LRC · 1 step-climb)
VIOLET = "#7C3AED"   # série 3 (ECON · 2 step-climbs)
GREEN  = "#059669"   # série 3 bis (CO, diagramme EI)
REDC   = "#E5342B"   # coût du temps (sémantique du cours)
BLUEC  = "#2F6BD8"   # coût total   (sémantique du cours)
GRAY   = "#8A919C"   # repères (MMO, annotations secondaires)

CMAP3D = LinearSegmentedColormap.from_list("blues_seq", ["#DCE8F8", "#1E3A8A"])

COL_W  = 3.5    # largeur d'une colonne IEEE [in]
FULL_W = 7.16   # largeur pleine page IEEE [in]

plt.rcParams.update({
    "font.family":      "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size":        8,
    "axes.labelsize":   8,
    "axes.titlesize":   8.5,
    "xtick.labelsize":  7.5,
    "ytick.labelsize":  7.5,
    "legend.fontsize":  7,
    "axes.linewidth":   0.6,
    "axes.edgecolor":   "#4B5563",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "grid.linewidth":   0.4,
    "grid.color":       "#D8DDE4",
    "lines.linewidth":  1.4,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype":     42,
})

KT = 0.514444
NM = 1852.0
FT = 0.3048

# --- Configuration de l'étude (paramètres communs à toutes les figures) ------
MASS_STUDY = 500_000.0     # masse de l'étude [kg]
ALT_STUDY  = 10_400.0      # altitude du point d'étude [m]  (≈ FL341)
CI_STUDY   = 180.0         # Cost Index [kg/min]  (choix justifié, cf. rapport)
BASE_EK    = 10_400.0      # altitude de base de la croisière EK215 [m]
DIST_EK    = 13_000e3      # distance de croisière EK215 [m]
MACH_EK    = 0.7476        # Mach économique à MASS_STUDY/ALT_STUDY/CI_STUDY (M_ECON)

VALS = {}   # chiffres exportés vers valeurs.json (cités dans le texte)


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    # Export SVG des figures 2D (revue visuelle Claude Design) ; les surfaces
    # 3D restent en PDF seul (SVG trop lourd, non éditable utilement).
    if "aero_surfaces" not in name and "geometry" not in name:
        fig.savefig(path.replace(".pdf", ".svg"))
    plt.close(fig)
    print(f"  → {name}")


# ---------------------------------------------------------------------------
# Figure 0a — profils atmosphériques ISA
# ---------------------------------------------------------------------------

def fig_atm():
    print("fig_atm …")
    h = np.linspace(0, 15_000, 300)
    T = mod_atm.temperature(h)
    P = mod_atm.pressure(h) / 1000.0
    rho = mod_atm.density(h)
    a = mod_atm.speed_of_sound(h)
    T_hot = mod_atm.temperature(h, 10.0)
    T_cold = mod_atm.temperature(h, -10.0)

    fig, axes = plt.subplots(2, 2, figsize=(COL_W, 3.6), sharey=True)
    specs = [(T, "$T$ [K]"), (P, "$P$ [kPa]"),
             (rho, r"$\rho$ [kg/m$^3$]"), (a, "$a$ [m/s]")]
    for ax, (y, lab) in zip(axes.flat, specs):
        ax.plot(y, h / 1000.0, color=INK, linewidth=1.3,
                label="ISA" if lab.startswith("$T$") else None)
        ax.axhline(11.0, color=GRAY, linewidth=0.7, linestyle=(0, (4, 3)))
        ax.set_xlabel(lab, labelpad=1)
        ax.grid(True, alpha=0.55)
    axes[0, 0].plot(T_hot, h / 1000.0, color=REDC, linewidth=0.9,
                    linestyle=(0, (4, 2)), label=r"$\Delta_{ISA}=+10$")
    axes[0, 0].plot(T_cold, h / 1000.0, color=BLUE, linewidth=0.9,
                    linestyle=(0, (4, 2)), label=r"$\Delta_{ISA}=-10$")
    axes[0, 0].legend(frameon=False, fontsize=5.6, loc="upper right",
                      handlelength=1.4, borderaxespad=0.1)
    axes[0, 1].annotate("tropopause — 11 km", (P[0] * 0.55, 11.25),
                        fontsize=6.0, color=GRAY)
    for ax in axes[:, 0]:
        ax.set_ylabel("Altitude [km]")
    fig.subplots_adjust(hspace=0.42, wspace=0.12)
    _save(fig, "fig_atm.pdf")


# ---------------------------------------------------------------------------
# Figure 0b — géométrie OpenVSP de l'avion (maillage .vspgeom)
# ---------------------------------------------------------------------------

def fig_geometry():
    print("fig_geometry …")
    fname = mod_aero.DEFAULT_FILE_GEOM
    with open(fname, "r") as f:
        tokens = f.read().split()
    idx = 0
    npt = int(tokens[idx]); idx += 1
    pt = np.array(tokens[idx:idx + 3 * npt], dtype=float
                  ).reshape((3, npt), order="F")
    idx += 3 * npt
    npoly = int(tokens[idx]); idx += 1
    con = np.array(tokens[idx:idx + 4 * npoly], dtype=int
                   ).reshape((4, npoly), order="F")[1:4, :] - 1

    fig = plt.figure(figsize=(COL_W, 2.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(pt[0], pt[1], pt[2], triangles=con.T,
                    color="#C9D7EC", edgecolor="#5B7DAE",
                    linewidth=0.05, alpha=0.95, rasterized=True)
    lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    ctr = lims.mean(axis=1)
    half = (lims[:, 1] - lims[:, 0]).max() / 2
    ax.set_xlim3d(ctr[0] - half * 0.72, ctr[0] + half * 0.72)
    ax.set_ylim3d(ctr[1] - half * 0.72, ctr[1] + half * 0.72)
    ax.set_zlim3d(ctr[2] - half * 0.38, ctr[2] + half * 0.38)
    ax.view_init(elev=18, azim=-125)
    ax.set_axis_off()
    fig.subplots_adjust(left=-0.25, right=1.25, bottom=-0.32, top=1.32)
    _save(fig, "fig_geometry.pdf")


# ---------------------------------------------------------------------------
# Figure 1 — surfaces aérodynamiques 3D (modèle spline évalué finement)
# ---------------------------------------------------------------------------

def fig_aero_surfaces(model):
    print("fig_aero_surfaces …")
    specs = [
        ("f_clwb", r"$C_{L_{wb}}$"), ("f_cdwb", r"$C_{D_{wb}}$"),
        ("f_cmwb", r"$C_{M_{wb}}$"),
        ("f_clht", r"$C_{L_{ht}}$"), ("f_cdht", r"$C_{D_{ht}}$"),
        ("f_cmht", r"$C_{M_{ht}}$"),
    ]
    fig = plt.figure(figsize=(FULL_W, 4.6))
    for i, (key, label) in enumerate(specs):
        grid = model[key]
        a = np.linspace(grid["x_alpha"][0], grid["x_alpha"][-1], 60)
        m = np.linspace(grid["y_mach"][0], grid["y_mach"][-1], 60)
        A, M = np.meshgrid(a, m)
        Z = grid["_interp"](a, m).T          # (n_m, n_a)
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        ax.plot_surface(A, M, Z, cmap=CMAP3D, rcount=40, ccount=40,
                        linewidth=0.1, edgecolor=(1, 1, 1, 0.25),
                        antialiased=True)
        ax.view_init(elev=22, azim=-135)
        ax.set_xlabel(r"$\alpha$ [$^\circ$]", labelpad=-4, fontsize=7)
        ax.set_ylabel("Mach", labelpad=-4, fontsize=7)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.zaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.tick_params(pad=-2, labelsize=6)
        ax.set_title(label, fontsize=9, pad=-2)
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.98,
                        wspace=0.08, hspace=0.06)
    _save(fig, "fig_aero_surfaces.pdf")


# ---------------------------------------------------------------------------
# Figure 1 bis — polaire et finesse de l'avion complet
# ---------------------------------------------------------------------------

def fig_polaire(model):
    print("fig_polaire …")
    alphas = np.linspace(-2.0, 12.0, 43)
    machs = [0.60, 0.70, 0.80]
    shades = ["#A9C4EC", "#5487D6", "#173F8A"]
    fig, (ax_p, ax_f) = plt.subplots(1, 2, figsize=(COL_W, 2.35))
    VALS["polaire"] = {}
    for M, shade in zip(machs, shades):
        cl = np.array([mod_aero.get_cl_total(model, float(a), M)
                       for a in alphas])
        cd = np.array([mod_aero.get_cd_total(model, float(a), M)
                       for a in alphas])
        ax_p.plot(cd, cl, color=shade, linewidth=1.2, label=f"M {M:.2f}")
        ld = np.where(cd > 0, cl / cd, np.nan)
        ax_f.plot(alphas, ld, color=shade, linewidth=1.2)
        i = int(np.nanargmax(ld))
        ax_p.plot(cd[i], cl[i], "*", color=AMBER, markersize=7, zorder=5)
        ax_f.plot(alphas[i], ld[i], "*", color=AMBER, markersize=7,
                  zorder=5)
        VALS["polaire"][f"{M:.2f}"] = {
            "ld_max": float(ld[i]), "alpha_ldmax": float(alphas[i]),
            "cl_ldmax": float(cl[i])}
    ax_p.set_xlabel("$C_{D_s}$")
    ax_p.set_ylabel("$C_{L_s}$")
    ax_p.grid(True, alpha=0.55)
    ax_p.legend(frameon=False, fontsize=6.2, loc="lower right",
                handlelength=1.4)
    ax_p.set_title("(a) Polaire", fontsize=7.5, loc="left")
    ax_f.set_xlabel(r"$\alpha$ [$^\circ$]")
    ax_f.set_ylabel("$C_{L_s}/C_{D_s}$", labelpad=1)
    ax_f.grid(True, alpha=0.55)
    ax_f.set_title("(b) Finesse", fontsize=7.5, loc="left")
    fig.subplots_adjust(wspace=0.34)
    _save(fig, "fig_polaire.pdf")


# ---------------------------------------------------------------------------
# Figure 10 bis — régime moteur et débit le long du vol EK215
# ---------------------------------------------------------------------------

def fig_ek_moteur(model):
    print("fig_ek_moteur …")
    cas = mod_traj.compare_step_climbs(_EK["mass0"], _EK["dist"], MACH_EK,
                                       _EK["base"], model=model)
    fig, (ax_n, ax_w) = plt.subplots(2, 1, figsize=(COL_W, 3.2),
                                     sharex=True)
    for k in (0, 1, 2):
        prof = cas[k]["result"]["profile"]
        s = np.array([p["s"] for p in prof]) / 1000.0
        n1 = np.array([p["N1"] for p in prof], dtype=float)
        wf = np.array([p["wf_kgh"] for p in prof], dtype=float) / 1000.0
        ax_n.plot(s, n1, color=_CASE_COL[k], linewidth=1.1,
                  label=_CASE_LAB[k])
        ax_w.plot(s, wf, color=_CASE_COL[k], linewidth=1.1)
    ax_n.set_ylabel("$N_1$ [\\%]")
    ax_n.grid(True, alpha=0.55)
    ax_n.legend(frameon=False, fontsize=6.4, loc="upper right", ncol=1)
    ax_w.set_ylabel("$W_{F_{tot}}$ [t/h]")
    ax_w.set_xlabel("Distance parcourue [km]")
    ax_w.grid(True, alpha=0.55)
    ax_w.set_xlim(0, 13_000)
    fig.subplots_adjust(hspace=0.13)
    _save(fig, "fig_ek_moteur.pdf")


# ---------------------------------------------------------------------------
# Figure 2 — diagramme ln-ln des indices d'émission de référence (BFF)
# ---------------------------------------------------------------------------

def fig_ei_lnln(model):
    print("fig_ei_lnln …")
    wf_ref = mod_prop._WF_C_REF
    series = [
        (mod_prop._EI_NOX_REF, "NO$_x$", BLUE,   "o"),
        (mod_prop._EI_CO_REF,  "CO",     GREEN,  "s"),
        (mod_prop._EI_UHC_REF, "UHC",    AMBER,  "D"),
    ]
    phases = ["Ralenti", "Approche", "Montée", "Décollage"]

    # Point de croisière de l'étude (500 t / base 10 400 m / M économique) :
    res = mod_trim.trim(MASS_STUDY, MACH_EK, BASE_EK, model=model)
    ei = mod_prop.get_emission_indices(res["N1"], MACH_EK, BASE_EK)
    wf_cruise = float(ei["WF_C_REF"])
    VALS["ei_cruise"] = {"N1": res["N1"], "WF_C_REF": wf_cruise,
                         "EI_NOx": float(ei["EI_NOx"]),
                         "EI_CO": float(ei["EI_CO"])}

    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    for y, lab, c, mk in series:
        ax.plot(wf_ref, y, color=c, marker=mk, markersize=4,
                markerfacecolor="white", markeredgewidth=1.1, label=lab)
    ax.axvline(wf_cruise, color=GRAY, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.text(wf_cruise * 1.06, 0.014, "croisière\n(débit corrigé réf.)",
            fontsize=6.5, color=GRAY, ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    secax = ax.secondary_xaxis("top")
    secax.set_xticks(list(wf_ref), labels=phases, rotation=22,
                     ha="left", fontsize=6.2)
    secax.tick_params(length=2, width=0.5, colors="#5A6472")
    secax.spines["top"].set_visible(False)
    for x in wf_ref:
        ax.axvline(x, color="#CBD2DA", linewidth=0.5,
                   linestyle=(0, (1, 2.5)), zorder=0)
    ax.set_xlabel(r"$W^{REF}_{F,C}$ [kg/s]  (échelle log)")
    ax.set_ylabel(r"$EI^{REF}_C$ [g/kg]  (échelle log)")
    ax.grid(True, which="both", alpha=0.55)
    ax.legend(frameon=False, loc="center left")
    _save(fig, "fig_ei_lnln.pdf")


# ---------------------------------------------------------------------------
# Figures 3 & 4 — SR(M) + optima et décomposition du coût (point de validation)
# ---------------------------------------------------------------------------

def fig_sr_and_cost(model):
    print("fig_sr_optima / fig_cout_ci …  (balayage du Mach)")
    res = mod_perf.cruise_speeds(MASS_STUDY, ALT_STUDY, cost_index=CI_STUDY,
                                 model=model)
    c = res["curve"]
    mach, sr, wf, tas, cost = (c["mach"], c["sr"], c["wf"], c["tas"], c["cost"])
    sr_nm = sr / NM

    opt = [(res["MRC"], "MRC", BLUE, "o"),
           (res["LRC"], "LRC", AMBER, "D"),
           (res["ECON"], "ECON", VIOLET, "s")]
    VALS["validation"] = {
        k: {"mach": res[k]["mach"], "tas_kt": res[k]["tas_kt"],
            "sr_nm_per_kg": res[k]["sr_nm_per_kg"],
            "wf_kgh": res[k]["wf_kgh"], "finesse": res[k]["finesse"],
            "N1": res[k]["N1"]}
        for k in ("MRC", "LRC", "ECON")}
    VALS["validation"]["sr_max_nm_per_kg"] = float(res["sr_max"] / NM)

    # --- SR(M) ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ax.plot(mach, sr_nm, color=INK, zorder=3)
    ax.axvline(0.89, color=GRAY, linewidth=0.8, linestyle=(0, (4, 3)))
    import matplotlib.transforms as mtransforms
    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(0.884, 0.97, "MMO 0,89", rotation=90, fontsize=6.5, color=GRAY,
            ha="right", va="top", transform=blend)
    place = {"MRC": ((-7, 6), "right", "bottom"),
             "LRC": ((7, 6), "left", "bottom"),
             "ECON": ((-8, -9), "right", "top")}
    for o, lab, col, mk in opt:
        y = o["sr_nm_per_kg"]
        ax.plot(o["mach"], y, mk, color=col, markersize=5.5,
                markerfacecolor="white", markeredgewidth=1.4, zorder=4)
        off, ha, va = place[lab]
        ax.annotate(f"{lab}\nM {o['mach']:.3f}".replace(".", ","),
                    (o["mach"], y), textcoords="offset points",
                    xytext=off, fontsize=6.8, color=col,
                    fontweight="bold", ha=ha, va=va)
    ax.set_xlabel("Nombre de Mach")
    ax.set_ylabel("Portée spécifique $SR$ [NM/kg]")
    ax.grid(True, alpha=0.55)
    _save(fig, "fig_sr_optima.pdf")

    # --- Décomposition du coût --------------------------------------------
    ci = CI_STUDY
    cost_fuel = wf / tas * NM                    # kg/NM
    cost_time = (ci / 60.0) / tas * NM           # kg/NM
    cost_tot  = cost * NM                        # kg/NM
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ax.plot(mach, cost_tot, color=BLUEC, label="Coût total", zorder=3)
    ax.plot(mach, cost_fuel, color=INK, linestyle=(0, (5, 2.5)),
            label="Coût carburant $W_F/V$")
    ax.plot(mach, cost_time, color=REDC, linestyle=(0, (1.5, 2)),
            label="Coût du temps $(\\beta\\,CI)/V$")
    m_mrc = res["MRC"]["mach"]
    m_eco = res["ECON"]["mach"]
    i_mrc = np.nanargmin(np.abs(mach - m_mrc))
    i_eco = np.nanargmin(np.abs(mach - m_eco))
    ax.plot(m_mrc, cost_fuel[i_mrc], "o", color=BLUE, markersize=5,
            markerfacecolor="white", markeredgewidth=1.3, zorder=4)
    ax.annotate("MRC", (m_mrc, cost_fuel[i_mrc]),
                textcoords="offset points", xytext=(2, -11),
                fontsize=6.8, color=BLUE, fontweight="bold")
    ax.plot(m_eco, cost_tot[i_eco], "s", color=VIOLET, markersize=5,
            markerfacecolor="white", markeredgewidth=1.3, zorder=4)
    ax.annotate("ECON", (m_eco, cost_tot[i_eco]),
                textcoords="offset points", xytext=(2, 6),
                fontsize=6.8, color=VIOLET, fontweight="bold")
    ax.axvline(0.89, color=GRAY, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.set_xlabel("Nombre de Mach")
    ax.set_ylabel("Coût par distance [kg/NM]")
    ax.grid(True, alpha=0.55)
    ax.legend(frameon=False, loc="upper center")
    _save(fig, "fig_cout_ci.pdf")


# ---------------------------------------------------------------------------
# Figure 5 — SR(M) pour plusieurs masses + lieux des optima
# ---------------------------------------------------------------------------

def fig_sr_masses(model):
    print("fig_sr_masses …  (4 masses × balayage)")
    masses = [400_000.0, 450_000.0, 500_000.0, 550_000.0]
    shades = ["#A9C4EC", "#6E9BDD", "#3A6FC9", "#173F8A"]  # séquentiel bleu
    fig, ax = plt.subplots(figsize=(COL_W, 2.75))
    loci = {"MRC": [], "LRC": [], "ECON": []}
    for mass, shade in zip(masses, shades):
        r = mod_perf.cruise_speeds(mass, ALT_STUDY, cost_index=CI_STUDY,
                                   model=model)
        c = r["curve"]
        ax.plot(c["mach"], c["sr"] / NM, color=shade, linewidth=1.2,
                label=f"{mass/1000:.0f} t")
        for k in loci:
            if r[k]:
                loci[k].append((r[k]["mach"], r[k]["sr_nm_per_kg"]))
        VALS.setdefault("multi_masses", {})[f"{mass/1000:.0f}"] = {
            k: (r[k]["mach"] if r[k] else None) for k in ("MRC", "LRC", "ECON")}
    marks = {"MRC": (BLUE, "o"), "LRC": (AMBER, "D"), "ECON": (VIOLET, "s")}
    for k, pts in loci.items():
        pts = np.array(pts)
        col, mk = marks[k]
        ax.plot(pts[:, 0], pts[:, 1], mk, color=col, markersize=4.5,
                markerfacecolor="white", markeredgewidth=1.2,
                linestyle=(0, (2, 2)), linewidth=0.8, label=k, zorder=4)
    ax.set_xlabel("Nombre de Mach")
    ax.set_ylabel("Portée spécifique $SR$ [NM/kg]")
    ax.grid(True, alpha=0.55)
    ax.legend(frameon=False, ncol=2, loc="lower center", columnspacing=1.0)
    _save(fig, "fig_sr_masses.pdf")


# ---------------------------------------------------------------------------
# Figure 6 — étude EK215 : profil vertical, masse, carburant, comparaison
# ---------------------------------------------------------------------------

def fig_ek215(model):
    print("fig_ek215 …  (3 trajectoires intégrées)")
    mass0 = MASS_STUDY
    dist = DIST_EK
    mach = MACH_EK
    base = BASE_EK
    cas = mod_traj.compare_step_climbs(mass0, dist, mach, base, model=model)

    _fl0 = round(BASE_EK / FT / 100)
    labels = {0: f"Direct FL{_fl0}", 1: "1 step-climb", 2: "2 step-climbs"}
    colors = {0: BLUE, 1: AMBER, 2: VIOLET}

    VALS["ek215"] = {}
    for k in (0, 1, 2):
        e = cas[k]
        VALS["ek215"][k] = {
            "feasible": e["feasible"],
            "time_h": e["time"] / 3600.0,
            "fuel_t": e["fuel"] / 1000.0,
            "levels_fl": [round(l / FT / 100) for l in e["levels"]],
            "emissions_kg": {p: e["emissions"][p] for p in e["emissions"]},
        }
    f0 = cas[0]["fuel"]
    for k in (1, 2):
        VALS["ek215"][k]["delta_fuel_pct"] = 100 * (cas[k]["fuel"] - f0) / f0
        VALS["ek215"][k]["delta_time_min"] = (cas[k]["time"]
                                              - cas[0]["time"]) / 60.0

    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 4.3))
    (ax_prof, ax_mass), (ax_fuel, ax_bar) = axes

    for k in (0, 1, 2):
        prof = cas[k]["result"]["profile"]
        s = np.array([p["s"] for p in prof]) / 1e6          # 1000 km
        alt = np.array([p["alt"] for p in prof]) / FT / 100  # FL
        m = np.array([p["mass"] for p in prof]) / 1000       # t
        burn = (mass0 - np.array([p["mass"] for p in prof])) / 1000
        ax_prof.plot(s * 1000, alt, color=colors[k], linewidth=1.3,
                     label=labels[k])
        ax_mass.plot(s * 1000, m, color=colors[k], linewidth=1.3)
        ax_fuel.plot(s * 1000, burn, color=colors[k], linewidth=1.3)

    _fls = [_fl0 + 20 * j for j in range(3)]     # FL341 / 361 / 381
    ax_prof.set_ylabel("Niveau de vol [FL]")
    ax_prof.set_ylim(_fls[0] - 6, _fls[-1] + 6)
    ax_prof.set_yticks(_fls)
    ax_prof.legend(frameon=False, loc="upper left")
    ax_prof.set_title("(a) Profil vertical", fontsize=8, loc="left")

    ax_mass.set_ylabel("Masse [t]")
    ax_mass.set_xlabel("Distance parcourue [km]")
    ax_mass.set_title("(b) Masse le long de la croisière", fontsize=8,
                      loc="left")

    ax_fuel.set_ylabel("Carburant brûlé [t]")
    ax_fuel.set_xlabel("Distance parcourue [km]")
    ax_fuel.set_title("(c) Carburant cumulé", fontsize=8, loc="left")

    fuels = [cas[k]["fuel"] / 1000 for k in (0, 1, 2)]
    xs = np.arange(3)
    bars = ax_bar.bar(xs, fuels, width=0.58,
                      color=[colors[k] for k in (0, 1, 2)], zorder=3)
    for x, f, k in zip(xs, fuels, (0, 1, 2)):
        top = f"{f:.1f} t".replace(".", ",")
        if k > 0:
            top += f"  ({VALS['ek215'][k]['delta_fuel_pct']:+.1f} %)".replace(
                ".", ",")
        ax_bar.annotate(top, (x, f), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7,
                        color=INK, fontweight="bold")
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels([labels[k] for k in (0, 1, 2)])
    ax_bar.set_ylabel("Carburant total [t]")
    ax_bar.set_ylim(0, max(fuels) * 1.16)
    ax_bar.set_title("(d) Consommation totale", fontsize=8, loc="left")

    for ax in (ax_prof, ax_mass, ax_fuel):
        ax.grid(True, alpha=0.55)
        ax.set_xlim(0, 13_000)
    ax_bar.grid(True, axis="y", alpha=0.55)
    ax_prof.set_xlabel("")
    ax_prof.tick_params(labelbottom=False)
    ax_bar.spines["left"].set_visible(True)

    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.09, top=0.94,
                        wspace=0.22, hspace=0.42)
    _save(fig, "fig_ek215.pdf")


# ---------------------------------------------------------------------------
# Figure 7 — influence de l'altitude : SR(M) par palier et SR_max(FL) par masse
# ---------------------------------------------------------------------------

FL2M = 100 * FT   # 1 FL = 100 ft


def fig_sr_altitudes(model):
    print("fig_sr_altitudes …  (balayages altitude × masse)")
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(COL_W, 4.7))

    # (a) SR(M) à 500 t pour quatre niveaux de vol.
    fls = [290, 310, 330, 350]
    shades = ["#A9C4EC", "#6E9BDD", "#3A6FC9", "#173F8A"]
    VALS["sr_vs_fl_500"] = {}
    for fl, shade in zip(fls, shades):
        r = mod_perf.cruise_speeds(MASS_STUDY, fl * FL2M, cost_index=CI_STUDY,
                                   model=model, refine=False)
        c = r["curve"]
        ax_a.plot(c["mach"], c["sr"] / NM, color=shade, linewidth=1.2,
                  label=f"FL{fl}")
        if r["MRC"]:
            ax_a.plot(r["MRC"]["mach"], r["MRC"]["sr_nm_per_kg"], "o",
                      color=BLUE, markersize=4, markerfacecolor="white",
                      markeredgewidth=1.1, zorder=4)
            VALS["sr_vs_fl_500"][fl] = {
                "mrc_mach": r["MRC"]["mach"],
                "sr_max_nm_per_kg": r["MRC"]["sr_nm_per_kg"]}
    ax_a.set_ylabel("$SR$ [NM/kg]")
    ax_a.set_xlabel("Nombre de Mach")
    ax_a.grid(True, alpha=0.55)
    ax_a.legend(frameon=False, ncol=2, loc="lower center",
                columnspacing=1.0, fontsize=6.6)
    ax_a.set_title("(a) $SR(M)$ à 500 t — effet du niveau de vol",
                   fontsize=8, loc="left")

    # (b) SR au Mach opérationnel 0,85 en fonction du niveau de vol, pour
    # trois masses : le palier qui maximise SR à M fixé monte quand la
    # masse diminue — le mécanisme du step-climb.
    fls_b = [290, 310, 330, 350, 370, 390]
    masses = [400_000.0, 450_000.0, 500_000.0]
    shades_b = ["#A9C4EC", "#5487D6", "#173F8A"]
    VALS["sr_mecon_vs_fl"] = {}
    for mass, shade in zip(masses, shades_b):
        pts = []
        for fl in fls_b:
            sr = mod_perf.specific_range(mass, MACH_EK, fl * FL2M, model=model)
            if np.isfinite(sr):
                pts.append((fl, sr / NM))
        pts = np.array(pts)
        VALS["sr_mecon_vs_fl"][f"{mass/1000:.0f}"] = {
            "fl": pts[:, 0].tolist(), "sr_nm_per_kg": pts[:, 1].tolist()}
        ax_b.plot(pts[:, 0], pts[:, 1], "-o", color=shade, linewidth=1.2,
                  markersize=3.4, markerfacecolor="white",
                  markeredgewidth=1.0, label=f"{mass/1000:.0f} t")
    ax_b.set_xlabel("Niveau de vol [FL]")
    ax_b.set_ylabel("$SR$ à $M_{ECON}$ [NM/kg]")
    ax_b.grid(True, alpha=0.55)
    ax_b.legend(frameon=False, loc="upper left", fontsize=6.6)
    ax_b.set_title("(b) $SR$ au Mach économique — monter "
                   "rapproche du régime optimal", fontsize=8, loc="left")
    fig.subplots_adjust(hspace=0.5)
    _save(fig, "fig_sr_altitudes.pdf")


# ---------------------------------------------------------------------------
# Figure 8 — résultats de l'équilibrage : α, δstab, F_N, W_F vs Mach
# ---------------------------------------------------------------------------

def fig_trim_analyse(model):
    print("fig_trim_analyse …")
    fls = [290, 330, 370]
    shades = ["#A9C4EC", "#5487D6", "#173F8A"]
    machs = np.linspace(0.55, 0.88, 23)
    panels = {"alpha": [], "dstab": [], "fn": [], "wf": []}
    data = {fl: {k: np.full(len(machs), np.nan) for k in panels}
            for fl in fls}
    for fl in fls:
        for i, M in enumerate(machs):
            try:
                res = mod_trim.trim(500_000.0, float(M), fl * FL2M,
                                    model=model)
            except ValueError:
                continue
            data[fl]["alpha"][i] = res["alpha"]
            data[fl]["dstab"][i] = res["dstab"]
            data[fl]["fn"][i] = res["FN"] / 1000.0            # kN
            if res["WF_total_kgh"]:
                data[fl]["wf"][i] = res["WF_total_kgh"] / 1000.0  # t/h

    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 1.95))
    specs = [("alpha", r"$\alpha$ [$^\circ$]"),
             ("dstab", r"$\delta_{stab}$ [$^\circ$]"),
             ("fn", r"$F_N$ totale [kN]"),
             ("wf", r"$W_{F_{tot}}$ [t/h]")]
    VALS["trim_vs_mach"] = {
        "mach": machs.tolist(),
        **{f"FL{fl}": {k: data[fl][k].tolist() for k in panels}
           for fl in fls}}
    for ax, (key, lab) in zip(axes, specs):
        for fl, shade in zip(fls, shades):
            ax.plot(machs, data[fl][key], color=shade, linewidth=1.2,
                    label=f"FL{fl}")
        ax.set_xlabel("Mach")
        ax.set_ylabel(lab, labelpad=1)
        ax.grid(True, alpha=0.55)
    axes[0].legend(frameon=False, fontsize=6.4, loc="best")
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.22, top=0.96,
                        wspace=0.36)
    _save(fig, "fig_trim_analyse.pdf")


# ---------------------------------------------------------------------------
# Figure 9 — influence du centrage sur le trim et la consommation
# ---------------------------------------------------------------------------

def fig_cg(model):
    print("fig_cg …")
    cgs = np.linspace(0.30, 0.45, 9)
    dstab = np.full(len(cgs), np.nan)
    wf = np.full(len(cgs), np.nan)
    for i, cg in enumerate(cgs):
        res = mod_trim.trim(MASS_STUDY, MACH_EK, BASE_EK, x_cg=float(cg),
                            model=model)
        dstab[i] = res["dstab"]
        if res["WF_total_kgh"]:
            wf[i] = res["WF_total_kgh"] / 1000.0
    VALS["cg_sweep"] = {"cg_pct": (cgs * 100).tolist(),
                        "dstab_deg": dstab.tolist(),
                        "wf_t_per_h": wf.tolist()}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 3.3), sharex=True)
    ax1.plot(cgs * 100, dstab, "-o", color=INK, markersize=3.4,
             markerfacecolor="white", markeredgewidth=1.0)
    ax1.axhline(0, color="#C6CCD4", linewidth=0.6)
    ax1.set_ylabel(r"$\delta_{stab}$ [$^\circ$]")
    ax1.grid(True, alpha=0.55)
    ax2.plot(cgs * 100, wf, "-o", color=BLUE, markersize=3.4,
             markerfacecolor="white", markeredgewidth=1.0)
    ax2.set_xlabel(r"Position du centre de gravité $x_{cg}$ [% MAC]")
    ax2.set_ylabel(r"$W_{F_{tot}}$ [t/h]")
    ax2.grid(True, alpha=0.55)
    for ax, cg_ref in ((ax1, 40), (ax2, 40)):
        ax.axvline(cg_ref, color=GRAY, linewidth=0.8, linestyle=(0, (4, 3)))
    ax1.set_title("500 t · FL341 · $M_{ECON}$ — centrage de référence 40 %",
                  fontsize=7.5, loc="left", color="#5A6472")
    fig.subplots_adjust(hspace=0.14)
    _save(fig, "fig_cg.pdf")


# ---------------------------------------------------------------------------
# Figure 9 bis — itinéraire du vol EK215 (contours réels Dubaï / Los Angeles)
# ---------------------------------------------------------------------------
# Contours géographiques simplifiés (source OpenStreetMap/Nominatim,
# polygon_geojson simplifié Douglas-Peucker, projection lon×cos(lat)),
# repris de la maquette Claude Design du projet. Coordonnées en unités SVG
# (repère 1000×248, y vers le bas), centrées sur chaque ville.

_DXB_PATH = ("M-32.0,0.4 L-20.9,11.6 L-10.8,16.6 L6.9,31.7 L9.1,31.7 "
             "L21.5,28.9 L25.9,25.2 L28.8,24.3 L32.0,17.0 L28.5,5.4 "
             "L28.1,-0.7 L22.3,-11.3 L10.3,-15.7 L9.4,-15.8 L8.5,-16.9 "
             "L8.8,-17.9 L5.1,-20.0 L-7.6,-31.7 L-11.4,-27.7 L-14.1,-23.2 "
             "L-18.3,-17.1 L-19.8,-11.8 L-23.0,-9.2 L-27.1,-5.6 L-30.2,-1.7 "
             "L-32.0,0.4 Z")
_LA_PATH = ("M-20.0,-17.7 L-16.5,-14.1 L-11.9,-12.4 L-13.8,-7.0 L-12.3,-6.3 "
            "L-12.3,1.0 L-9.8,-0.0 L-6.3,-4.8 L-2.6,-1.5 L-9.4,3.6 L-1.2,7.8 "
            "L-0.6,7.9 L-1.1,6.5 L3.6,6.5 L3.3,4.4 L2.8,3.9 L3.7,1.7 "
            "L6.2,2.5 L7.6,2.7 L8.3,5.8 L9.7,6.5 L9.7,12.7 L8.3,13.1 "
            "L8.1,22.5 L8.5,23.0 L5.4,32.0 L12.6,30.9 L14.4,27.1 L13.2,23.1 "
            "L15.3,20.6 L14.8,16.3 L13.9,18.9 L9.1,18.8 L9.7,14.6 L10.5,12.1 "
            "L10.4,7.3 L14.5,6.7 L13.4,4.4 L12.7,5.3 L13.3,0.9 L13.9,0.8 "
            "L14.5,-1.5 L17.6,-1.5 L17.6,-5.9 L19.7,-6.0 L20.0,-9.5 "
            "L18.6,-10.6 L19.6,-12.1 L18.3,-12.4 L17.8,-14.3 L14.8,-14.2 "
            "L14.1,-12.1 L12.5,-11.5 L10.4,-15.0 L8.2,-15.4 L5.4,-13.8 "
            "L3.8,-18.7 L3.5,-19.2 L5.9,-20.1 L9.6,-21.2 L11.6,-22.0 "
            "L11.8,-23.9 L11.1,-26.9 L10.1,-27.9 L9.1,-27.3 L3.7,-26.8 "
            "L2.3,-27.5 L1.6,-28.5 L0.5,-27.4 L1.4,-30.5 L0.5,-31.5 "
            "L-7.3,-32.0 L-9.7,-30.1 L-10.4,-30.1 L-10.2,-28.5 L-12.3,-28.2 "
            "L-14.0,-27.5 L-14.9,-26.5 L-17.3,-25.7 L-17.1,-22.6 "
            "L-18.4,-22.6 L-19.3,-20.7 L-19.3,-18.6 L-19.3,-18.1 "
            "L-20.0,-17.7 Z")


def _svg_pts(path, tx, ty, sc):
    """Convertit un path SVG (M/L absolus) en tableau (x, y) matplotlib."""
    import re
    pts = [(float(a), float(b)) for a, b in
           re.findall(r"(-?\d+\.?\d*),(-?\d+\.?\d*)", path)]
    return np.array([(tx + sc * x, -(ty + sc * y)) for x, y in pts])


def fig_ek_route():
    print("fig_ek_route …")
    from matplotlib.patches import Polygon
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    ax.set_axis_off()

    for path, tx, ty, sc in ((_DXB_PATH, 150, 168, 0.85),
                             (_LA_PATH, 850, 146, 0.95)):
        pts = _svg_pts(path, tx, ty, sc)
        ax.add_patch(Polygon(pts, closed=True, facecolor="#CBDCF3",
                             edgecolor=BLUE, linewidth=0.9, zorder=2))

    # Arc grand cercle (Bézier quadratique M150,168 Q500,38 850,148).
    t = np.linspace(0, 1, 120)
    p0, p1, p2 = np.array([150, -168.]), np.array([500, -38.]), \
        np.array([850, -148.])
    arc = ((1 - t)[:, None] ** 2 * p0 + 2 * ((1 - t) * t)[:, None] * p1
           + (t ** 2)[:, None] * p2)
    ax.plot(arc[:, 0], arc[:, 1], color=INK, linewidth=1.1,
            linestyle=(0, (5, 4)), zorder=3)
    for x, y in ((150, -168), (850, -148)):
        ax.plot(x, y, "o", color=INK, markersize=4.5,
                markerfacecolor="white", markeredgewidth=1.3, zorder=4)

    ax.annotate("Dubaï — DXB\n(départ)", (150, -205), ha="center",
                va="top", fontsize=7, color=INK, fontweight="bold")
    ax.annotate("Los Angeles — LAX\n(arrivée)", (850, -205), ha="center",
                va="top", fontsize=7, color=INK, fontweight="bold")
    ax.annotate("grand cercle ≈ 13 400 km — croisière étudiée 13 000 km",
                (500, -66), ha="center", va="bottom",
                fontsize=6.8, color="#5A6472")

    ax.set_xlim(60, 940)
    ax.set_ylim(-252, -30)
    ax.set_aspect("equal")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    _save(fig, "fig_ek_route.pdf")


# ---------------------------------------------------------------------------
# Figures 10-12 — sensibilités de l'étude EK215 (Mach, palier de base, ΔISA)
# ---------------------------------------------------------------------------

_EK = dict(mass0=MASS_STUDY, dist=DIST_EK, base=BASE_EK)
_CASE_COL = {0: BLUE, 1: AMBER, 2: VIOLET}
_CASE_LAB = {0: "Direct", 1: "1 step-climb", 2: "2 step-climbs"}


def _ek_fuels(model, mach=MACH_EK, base=None, disa=0.0):
    base = _EK["base"] if base is None else base
    cas = mod_traj.compare_step_climbs(_EK["mass0"], _EK["dist"], mach, base,
                                       delta_isa=disa, model=model)
    return {k: (cas[k]["fuel"] / 1000.0 if cas[k]["feasible"] else np.nan)
            for k in (0, 1, 2)}


def _ek_emissions(model, mach=MACH_EK, base=None):
    """Masses de polluants {cas: {pol: kg}} pour un point de la comparaison."""
    base = _EK["base"] if base is None else base
    cas = mod_traj.compare_step_climbs(_EK["mass0"], _EK["dist"], mach, base,
                                       model=model)
    out = {}
    for k in (0, 1, 2):
        if cas[k]["feasible"]:
            out[k] = dict(cas[k]["emissions"])
        else:
            out[k] = {p: np.nan for p in ("NOx", "UHC", "CO", "CO2")}
    return out


_POL_PANELS = [("CO2", "CO$_2$ [t]", 1e-3), ("NOx", "NO$_x$ [kg]", 1.0),
               ("CO", "CO [kg]", 1.0), ("UHC", "UHC [kg]", 1.0)]


def _fig_emissions_grid(xvals, emis_by_x, xlabel, fname, vline=None):
    """Grille 2×2 CO2/NOx/CO/UHC, une courbe par stratégie 0/1/2."""
    fig, axes = plt.subplots(2, 2, figsize=(COL_W, 3.7))
    for ax, (pol, lab, fac) in zip(axes.flat, _POL_PANELS):
        for k in (0, 1, 2):
            y = [emis_by_x[i][k][pol] * fac for i in range(len(xvals))]
            ax.plot(xvals, y, "-o", color=_CASE_COL[k], linewidth=1.1,
                    markersize=2.8, markerfacecolor="white",
                    markeredgewidth=0.9,
                    label=_CASE_LAB[k] if pol == "CO2" else None)
        if vline is not None:
            ax.axvline(vline, color=GRAY, linewidth=0.7,
                       linestyle=(0, (4, 3)))
        ax.set_ylabel(lab, labelpad=1)
        ax.grid(True, alpha=0.55)
        ax.tick_params(labelsize=6.6)
    for ax in axes[1]:
        ax.set_xlabel(xlabel)
    for ax in axes[0]:
        ax.tick_params(labelbottom=False)
    axes[0, 0].legend(frameon=False, fontsize=5.8, loc="best",
                      handlelength=1.4)
    fig.subplots_adjust(hspace=0.14, wspace=0.42)
    _save(fig, fname)


def fig_ek_emissions_sweeps(model):
    print("fig_ek_emis_mach …  (Mach 0,40–0,90 × 3 cas)")
    machs = [round(x, 2) for x in np.arange(0.40, 0.905, 0.05)]
    emis_m = [_ek_emissions(model, mach=M) for M in machs]
    VALS["ek_emis_mach"] = {
        "mach": machs,
        **{str(k): {p: [e[k][p] for e in emis_m]
                    for p in ("NOx", "UHC", "CO", "CO2")}
           for k in (0, 1, 2)}}
    _fig_emissions_grid(machs, emis_m, "Mach de croisière",
                        "fig_ek_emis_mach.pdf", vline=MACH_EK)

    print("fig_ek_dh …  (amplitudes × 3 cas)")
    steps_ft = [500, 1500, 2500, 3500, 4500, 5500, 6500, 7500]
    fuels_dh = {k: [] for k in (0, 1, 2)}
    for sft in steps_ft:
        cas = mod_traj.compare_step_climbs(_EK["mass0"], _EK["dist"], MACH_EK,
                                           _EK["base"], step_ft=float(sft),
                                           model=model)
        for k in (0, 1, 2):
            fuels_dh[k].append(cas[k]["fuel"] / 1000.0
                               if cas[k]["feasible"] else np.nan)
    VALS["ek_vs_dh"] = {"step_ft": steps_ft,
                        **{str(k): fuels_dh[k] for k in (0, 1, 2)}}

    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    for k in (0, 1, 2):
        ax.plot(steps_ft, fuels_dh[k], "-o", color=_CASE_COL[k],
                linewidth=1.2, markersize=3.4, markerfacecolor="white",
                markeredgewidth=1.0, label=_CASE_LAB[k])
    ax.axvline(2000, color=GRAY, linewidth=0.8, linestyle=(0, (4, 3)))
    import matplotlib.transforms as mtransforms
    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(2030, 0.03, "RVSM 2 000 ft", rotation=90, fontsize=6.3,
            color=GRAY, ha="left", va="bottom", transform=blend)
    ax.set_xlabel(r"Amplitude d'un step-climb $\Delta h$ [ft]")
    ax.set_ylabel("Carburant total [t]")
    ax.set_xticks(steps_ft)
    ax.tick_params(axis="x", labelsize=6.6)
    ax.grid(True, alpha=0.55)
    ax.legend(frameon=False, fontsize=6.6, loc="upper center")
    _save(fig, "fig_ek_dh.pdf")

    print("fig_ek_emis_alt …  (altitude 7000–14500 m × 3 cas, M0,75)")
    alts_m = list(range(7000, 14501, 500))         # base de croisière [m]
    emis_a = []
    for a in alts_m:
        cas = mod_traj.compare_step_climbs(_EK["mass0"], _EK["dist"], 0.75,
                                           float(a), model=model)
        emis_a.append({k: (dict(cas[k]["emissions"]) if cas[k]["feasible"]
                           else {p: np.nan for p in
                                 ("NOx", "UHC", "CO", "CO2")})
                       for k in (0, 1, 2)})
    VALS["ek_emis_alt"] = {
        "alt_m": alts_m,
        **{str(k): {p: [e[k][p] for e in emis_a]
                    for p in ("NOx", "UHC", "CO", "CO2")}
           for k in (0, 1, 2)}}
    _fig_emissions_grid([a / 1000 for a in alts_m], emis_a,
                        "Altitude de base [km]", "fig_ek_emis_alt.pdf")

    print("fig_ek_emis_base …  (5 paliers × 3 cas)")
    bases = [321, 331, 341, 351, 361]
    emis_b = [_ek_emissions(model, base=fl * FL2M) for fl in bases]
    VALS["ek_emis_base"] = {
        "fl": bases,
        **{str(k): {p: [e[k][p] for e in emis_b]
                    for p in ("NOx", "UHC", "CO", "CO2")}
           for k in (0, 1, 2)}}
    _fig_emissions_grid(bases, emis_b, "Palier initial [FL]",
                        "fig_ek_emis_base.pdf", vline=341)


def fig_ek_sensibilites(model):
    print("fig_ek_mach …  (Mach × 3 cas)")
    machs = [0.64, 0.68, 0.72, 0.76, 0.80, 0.84, 0.88]
    fuels_m = {k: [] for k in (0, 1, 2)}
    for M in machs:
        f = _ek_fuels(model, mach=M)
        for k in (0, 1, 2):
            fuels_m[k].append(f[k])
    VALS["ek_vs_mach"] = {"mach": machs,
                          **{str(k): fuels_m[k] for k in (0, 1, 2)}}

    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    for k in (0, 1, 2):
        ax.plot(machs, fuels_m[k], "-o", color=_CASE_COL[k], linewidth=1.2,
                markersize=3.4, markerfacecolor="white",
                markeredgewidth=1.0, label=_CASE_LAB[k])
    ax.axvline(MACH_EK, color=GRAY, linewidth=0.8, linestyle=(0, (4, 3)))
    import matplotlib.transforms as mtransforms
    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(MACH_EK + 0.002, 0.03, "$M_{ECON}$", rotation=90, fontsize=6.6,
            color=GRAY, ha="left", va="bottom", transform=blend)
    ax.set_xlabel("Mach de croisière")
    ax.set_ylabel("Carburant total [t]")
    ax.grid(True, alpha=0.55)
    ax.legend(frameon=False, fontsize=6.6, loc="upper left")
    _save(fig, "fig_ek_mach.pdf")

    print("fig_ek_base …  (5 paliers × 3 cas)")
    bases = [321, 331, 341, 351, 361]
    fuels_b = {k: [] for k in (0, 1, 2)}
    for fl in bases:
        f = _ek_fuels(model, base=fl * FL2M)
        for k in (0, 1, 2):
            fuels_b[k].append(f[k])
    VALS["ek_vs_base"] = {"fl": bases,
                          **{str(k): fuels_b[k] for k in (0, 1, 2)}}

    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    for k in (0, 1, 2):
        ax.plot(bases, fuels_b[k], "-o", color=_CASE_COL[k], linewidth=1.2,
                markersize=3.4, markerfacecolor="white",
                markeredgewidth=1.0, label=_CASE_LAB[k])
    ax.axvline(341, color=GRAY, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.set_xlabel("Palier initial [FL]")
    ax.set_ylabel("Carburant total [t]")
    ax.set_xticks(bases)
    ax.grid(True, alpha=0.55)
    ax.legend(frameon=False, fontsize=6.6, loc="upper right")
    _save(fig, "fig_ek_base.pdf")

    print("fig_ek_disa …  (5 ΔISA × 3 cas)")
    disas = [-10.0, -5.0, 0.0, 5.0, 10.0]
    fuels_d = {k: [] for k in (0, 1, 2)}
    for d in disas:
        f = _ek_fuels(model, disa=d)
        for k in (0, 1, 2):
            fuels_d[k].append(f[k])
    VALS["ek_vs_disa"] = {"disa": disas,
                          **{str(k): fuels_d[k] for k in (0, 1, 2)}}

    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    for k in (0, 1, 2):
        ax.plot(disas, fuels_d[k], "-o", color=_CASE_COL[k], linewidth=1.2,
                markersize=3.4, markerfacecolor="white",
                markeredgewidth=1.0, label=_CASE_LAB[k])
    ax.set_xlabel(r"Déviation de température $\Delta_{ISA}$ [$^\circ$C]")
    ax.set_ylabel("Carburant total [t]")
    ax.set_xticks(disas)
    ax.grid(True, alpha=0.55)
    ax.legend(frameon=False, fontsize=6.6, loc="upper right")
    _save(fig, "fig_ek_disa.pdf")


def fig_ek_wind(model):
    """Carburant EK215 en fonction d'un vent longitudinal constant (V_GS =
    V_TAS ± V_W), pour les trois stratégies de paliers."""
    print("fig_ek_wind …  (9 vents × 3 cas)")
    winds = [-100.0, -75.0, -50.0, -25.0, 0.0, 25.0, 50.0, 75.0, 100.0]
    fuels_w = {k: [] for k in (0, 1, 2)}
    for w in winds:
        cas = mod_traj.compare_step_climbs(_EK["mass0"], _EK["dist"], MACH_EK,
                                           _EK["base"], model=model, wind_kt=w)
        for k in (0, 1, 2):
            fuels_w[k].append(cas[k]["fuel"] / 1000.0
                              if cas[k]["feasible"] else np.nan)
    VALS["ek_vs_wind"] = {"wind_kt": winds,
                          **{str(k): fuels_w[k] for k in (0, 1, 2)}}

    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    for k in (0, 1, 2):
        ax.plot(winds, fuels_w[k], "-o", color=_CASE_COL[k], linewidth=1.2,
                markersize=3.4, markerfacecolor="white",
                markeredgewidth=1.0, label=_CASE_LAB[k])
    ax.axvline(0.0, color=GRAY, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.set_xlabel("Vent longitudinal [kt]  (>0 : vent arrière)")
    ax.set_ylabel("Carburant total [t]")
    ax.set_xticks(winds[::2])
    ax.grid(True, alpha=0.55)
    ax.legend(frameon=False, fontsize=6.6, loc="upper right")
    _save(fig, "fig_ek_wind.pdf")


# ---------------------------------------------------------------------------
# Figure 13 — influence du Cost Index sur le Mach ECON
# ---------------------------------------------------------------------------

def fig_econ_ci(model):
    print("fig_econ_ci …  (7 Cost Index)")
    cis = [0, 50, 100, 180, 250, 350, 500]
    shades = ["#BBD2F0", "#9CBBE8", "#7BA2DE", "#5B88D2",
              "#3D6EC2", "#2453A4", "#123A7E"]
    base = mod_perf.cruise_speeds(MASS_STUDY, ALT_STUDY, cost_index=CI_STUDY,
                                  model=model)
    c = base["curve"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ax.plot(c["mach"], c["sr"] / NM, color=INK, zorder=2)
    VALS["econ_vs_ci"] = {}
    for ci, shade in zip(cis, shades):
        r = mod_perf.cruise_speeds(MASS_STUDY, ALT_STUDY, cost_index=float(ci),
                                   model=model)
        if r["ECON"]:
            m, s = r["ECON"]["mach"], r["ECON"]["sr_nm_per_kg"]
            VALS["econ_vs_ci"][ci] = m
            ax.plot(m, s, "s", color=shade, markersize=4.8,
                    markerfacecolor=shade, markeredgecolor="white",
                    markeredgewidth=0.9, zorder=4)
            if ci in (0, 180, 500):
                ax.annotate(f"CI {ci}", (m, s), textcoords="offset points",
                            xytext=(5, 5), fontsize=6.4, color=shade,
                            fontweight="bold")
    ax.plot(base["MRC"]["mach"], base["MRC"]["sr_nm_per_kg"], "o",
            color=BLUE, markersize=5.5, markerfacecolor="white",
            markeredgewidth=1.4, zorder=3)
    ax.annotate("MRC", (base["MRC"]["mach"], base["MRC"]["sr_nm_per_kg"]),
                textcoords="offset points", xytext=(-6, 6), fontsize=6.8,
                color=BLUE, fontweight="bold", ha="right")
    ax.set_xlabel("Nombre de Mach")
    ax.set_ylabel("Portée spécifique $SR$ [NM/kg]")
    ax.grid(True, alpha=0.55)
    _save(fig, "fig_econ_ci.pdf")


# ---------------------------------------------------------------------------
# Figure — ce que le Cost Index achète : temps de vol contre carburant
# ---------------------------------------------------------------------------

def fig_ci_temps(model):
    """Temps de croisière et carburant du vol EK215 en fonction du Cost Index.

    Le CI ne modifie pas la trajectoire : il déplace le Mach économique, et
    c'est ce Mach qui fixe le temps et la consommation. La figure montre donc
    le compromis que l'on achète en montant le CI — et situe la valeur retenue
    pour l'étude.
    """
    cis = [0, 25, 50, 100, 150, 180, 250, 320, 400, 500]
    print(f"fig_ci_temps …  ({len(cis)} Cost Index × un vol EK215 complet)")

    machs, temps, carbu = [], [], []
    for ci in cis:
        r = mod_perf.cruise_speeds(MASS_STUDY, ALT_STUDY, cost_index=float(ci),
                                   model=model)
        if not r["ECON"]:
            continue
        m = r["ECON"]["mach"]
        vol = mod_traj.integrate_segment(MASS_STUDY, DIST_EK, BASE_EK, m,
                                         model=model)
        machs.append(m)
        temps.append(vol["time"] / 3600.0)
        carbu.append(vol["fuel"] / 1000.0)
        print(f"     CI {ci:3d} → M {m:.3f} · {temps[-1]:.2f} h · {carbu[-1]:.1f} t")

    cis_ok = [c for c in cis][:len(machs)]
    VALS["ci_temps"] = {
        str(c): {"mach": m, "time_h": t, "fuel_t": f}
        for c, m, t, f in zip(cis_ok, machs, temps, carbu)}

    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(COL_W, 3.5))

    # (a) temps de croisière — rouge : sémantique « coût du temps » du cours
    a1.plot(cis_ok, temps, color=REDC, marker="o", markersize=3.4,
            markerfacecolor="white", markeredgewidth=1.0, zorder=3)
    a1.set_ylabel("Temps de croisière [h]")
    a1.set_title("(a)", loc="left", fontsize=8, color=GRAY)

    # (b) carburant — encre : la contrepartie payée
    a2.plot(cis_ok, carbu, color=INK, marker="o", markersize=3.4,
            markerfacecolor="white", markeredgewidth=1.0, zorder=3)
    a2.set_ylabel("Carburant [t]")
    a2.set_xlabel("Cost Index [kg/min]")
    a2.set_title("(b)", loc="left", fontsize=8, color=GRAY)

    i180 = cis_ok.index(int(CI_STUDY))
    for ax, serie, col in ((a1, temps, REDC), (a2, carbu, INK)):
        ax.axvline(CI_STUDY, color=GRAY, linestyle="--", linewidth=0.8, zorder=1)
        ax.plot(CI_STUDY, serie[i180], "s", color=col, markersize=5.2,
                markeredgecolor="white", markeredgewidth=1.0, zorder=4)
        ax.grid(True, alpha=0.55)
    a1.annotate(f"CI retenu = {CI_STUDY:.0f}\n"
                f"$M^{{ECON}}$ = {machs[i180]:.3f}".replace(".", ","),
                (CI_STUDY, temps[i180]), textcoords="offset points",
                xytext=(9, 9), fontsize=6.6, color=GRAY)
    # ce que coûte et ce que rapporte le passage de CI 0 à CI 500, posé dans le
    # coin laissé libre par la courbe (montante en (b), descendante en (a))
    a1.text(0.035, 0.07, f"CI 0 → 500 : −{(temps[0] - temps[-1]) * 60:.0f} min",
            transform=a1.transAxes, fontsize=6.6, color=REDC, fontweight="bold")
    a2.text(0.035, 0.87,
            f"CI 0 → 500 : +{carbu[-1] - carbu[0]:.1f} t".replace(".", ","),
            transform=a2.transAxes, fontsize=6.6, color=INK, fontweight="bold")
    _save(fig, "fig_ci_temps.pdf")


# ---------------------------------------------------------------------------
# Figure 14 — évolution relative des émissions selon la stratégie de paliers
# ---------------------------------------------------------------------------

def fig_ek_emissions(model=None):
    """Barres d'écart relatif vs vol direct (utilise VALS['ek215'])."""
    print("fig_ek_emissions …")
    ek = VALS["ek215"]
    pols = [("Carburant", "fuel_t", None),
            ("CO$_2$", None, "CO2"),
            ("NO$_x$", None, "NOx"),
            ("CO", None, "CO"),
            ("UHC", None, "UHC")]
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    width = 0.34
    xs = np.arange(len(pols))
    for j, k in enumerate((1, 2)):
        deltas = []
        for _, key_f, key_e in pols:
            if key_f:
                d = 100 * (ek[k][key_f] - ek[0][key_f]) / ek[0][key_f]
            else:
                d = 100 * (ek[k]["emissions_kg"][key_e]
                           - ek[0]["emissions_kg"][key_e]) \
                    / ek[0]["emissions_kg"][key_e]
            deltas.append(d)
        off = (j - 0.5) * width
        color = _CASE_COL[k]
        bars = ax.bar(xs + off, deltas, width=width - 0.03, color=color,
                      zorder=3, label=_CASE_LAB[k])
        for x, d in zip(xs + off, deltas):
            dy = 4 if d >= 0 else -9
            ax.annotate(f"{d:+.1f}".replace(".", ","), (x, d),
                        textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=6.2, color=color,
                        fontweight="bold")
    ax.axhline(0, color="#4B5563", linewidth=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in pols])
    ax.set_ylabel("Écart au vol direct [%]")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 2, hi + 3)
    ax.grid(True, axis="y", alpha=0.55)
    ax.legend(frameon=False, fontsize=6.6, loc="lower left")
    _save(fig, "fig_ek_emissions.pdf")


# ---------------------------------------------------------------------------
# Figure — poussée requise en fonction de l'altitude (au Mach économique)
# ---------------------------------------------------------------------------
# Illustre le principe du step-climb : à mesure que l'avion s'allège, la cuvette
# de poussée requise se creuse et son minimum se décale vers le HAUT ; l'avion
# « descend » d'une cuvette à l'autre en montant de palier. Les trois masses sont
# celles de l'avion au moment où il exécute le 0ᵉ / 1ᵉ / 2ᵉ step-climb dans le
# scénario 2 step-climbs de l'étude EK215.

def fig_thrust_altitude(model):
    print("fig_thrust_altitude …")
    mach = MACH_EK

    # Masses au début de chaque palier du cas 2 step-climbs (= masse à laquelle
    # l'avion effectue le k-ième step-climb).
    cas = mod_traj.compare_step_climbs(MASS_STUDY, DIST_EK, mach, BASE_EK,
                                       model=model)
    prof = cas[2]["result"]["profile"]
    masses, last = [], None
    for s in prof:
        if s["alt"] != last:
            masses.append(s["mass"])
            last = s["alt"]
    masses = masses[:3]                     # 3 paliers -> 3 masses

    labels = {0: "0 step-climb", 1: "1 step-climb", 2: "2 step-climbs"}
    colors = {0: BLUE, 1: AMBER, 2: VIOLET}

    alts = np.linspace(6_000.0, 15_000.0, 46)   # 6 -> 15 km

    fig, ax = plt.subplots(figsize=(COL_W, 2.7))
    VALS["thrust_altitude"] = {"mach": mach}
    for k, m in enumerate(masses):
        fn = np.full_like(alts, np.nan)
        for i, h in enumerate(alts):
            try:
                r = mod_trim.trim(float(m), mach, float(h), model=model)
                if r["FN"] is not None:
                    fn[i] = r["FN"] / 1000.0            # kN (total 4 moteurs)
            except ValueError:
                pass
        ax.plot(alts / 1000.0, fn, color=colors[k], linewidth=1.4,
                label=f"{labels[k]}  ({m/1000:.0f} t)".replace(".", ","))
        # marqueur du minimum (altitude optimale à cette masse)
        if np.any(np.isfinite(fn)):
            j = int(np.nanargmin(fn))
            ax.plot(alts[j] / 1000.0, fn[j], "o", color=colors[k],
                    markersize=4.5, markerfacecolor="white",
                    markeredgewidth=1.2, zorder=5)
            VALS["thrust_altitude"][k] = {
                "mass_t": m / 1000.0,
                "alt_opt_m": float(alts[j]),
                "fn_min_kN": float(fn[j]),
            }

    ax.set_xlabel("Altitude [km]")
    ax.set_ylabel("Poussée requise [kN]")
    ax.grid(True, alpha=0.55)
    ax.legend(frameon=False, loc="upper center", fontsize=6.6,
              handlelength=1.6)
    _save(fig, "fig_thrust_altitude.pdf")


# ---------------------------------------------------------------------------
# Chiffres complémentaires cités dans le texte
# ---------------------------------------------------------------------------

def extra_values(model):
    print("valeurs complémentaires …")
    # Point vitrine de trim au point d'étude (convergence, section V-A).
    res = mod_trim.trim(MASS_STUDY, MACH_EK, BASE_EK, model=model)
    VALS["trim_vitrine"] = {
        "alpha": res["alpha"], "dstab": res["dstab"],
        "FN_kN": res["FN"] / 1000.0, "N1": res["N1"],
        "WF_total_kgh": res["WF_total_kgh"], "finesse": res["finesse"],
        "iterations": res["iterations"], "converged": res["converged"],
        "history": [
            {"it": h["it"], "alpha": h["alpha"], "dstab": h["dstab"],
             "FN_kN": h["FN"] / 1000.0,
             "d_alpha": h["d_alpha"], "d_FN": h["d_FN"],
             "d_dstab": h["d_dstab"]}
            for h in res["history"]],
    }
    # Atmosphère au point de validation (10 668 m).
    VALS["atm_10668"] = {
        "T": float(mod_atm.temperature(10_668.0)),
        "P": float(mod_atm.pressure(10_668.0)),
        "rho": float(mod_atm.density(10_668.0)),
        "a": float(mod_atm.speed_of_sound(10_668.0)),
    }


def extra_revue(model):
    """Valeurs des analyses ajoutées en revue : Mach ECON adaptatif par palier,
    migration du centrage pendant le vol, coût énergétique d'un step-climb."""
    print("ECON adaptatif …  (3 cas)")
    step = mod_traj.STEP_CLIMB_FT * FT
    VALS["ek_econ_adaptatif"] = {}
    for k in (0, 1, 2):
        levels = [_EK["base"] + j * step for j in range(k + 1)]
        # Cas Mach fixe (référence de coût) :
        rf = mod_traj.fly(_EK["mass0"],
                          [(_EK["dist"] / len(levels), a, MACH_EK)
                           for a in levels], model=model)
        # Cas adaptatif : M_ECON recalculé au début de chaque palier.
        ra = mod_traj.fly_econ(_EK["mass0"],
                               [(_EK["dist"] / len(levels), a) for a in levels],
                               CI_STUDY, model=model)
        cost = lambda r: (r["fuel"] + CI_STUDY * r["time"] / 60.0) / 1000.0
        VALS["ek_econ_adaptatif"][str(k)] = {
            "machs":        [float(m) for m in ra["machs"]],
            "fuel_t":       ra["fuel"] / 1000.0,
            "time_h":       ra["time"] / 3600.0,
            "cost_t":       cost(ra),
            "fixe_fuel_t":  rf["fuel"] / 1000.0,
            "fixe_time_h":  rf["time"] / 3600.0,
            "fixe_cost_t":  cost(rf),
        }

    print("migration du centrage …  (2 migrations × 2 cas)")
    VALS["ek_cg_migration"] = {}
    for k in (0, 2):
        levels = [_EK["base"] + j * step for j in range(k + 1)]
        segs = [(_EK["dist"] / len(levels), a, MACH_EK) for a in levels]
        ref = mod_traj.fly(_EK["mass0"], segs, model=model)   # CG figé 0.40
        entry = {"fixe_fuel_t": ref["fuel"] / 1000.0}
        for x1, lab in ((0.36, "avant_36"), (0.44, "arriere_44")):
            scale = ref["fuel"]   # migration linéaire sur le carburant du vol
            cg = (lambda m, x1=x1, scale=scale:
                  0.40 + (x1 - 0.40) * min(1.0, (_EK["mass0"] - m) / scale))
            r = mod_traj.fly(_EK["mass0"], segs, model=model, x_cg=cg)
            entry[lab] = {
                "fuel_t":    r["fuel"] / 1000.0,
                "delta_pct": 100.0 * (r["fuel"] - ref["fuel"]) / ref["fuel"],
            }
        VALS["ek_cg_migration"][str(k)] = entry

    # Comparaison à la littérature : vol direct à M0,85 (Mach opérationnel,
    # Airbus Facts & Figures) → consommation kilométrique et par siège
    # (545 sièges, configuration 4 classes typique Airbus).
    print("comparaison littérature …  (direct M0.85)")
    seats = 545.0
    r085 = mod_traj.fly(_EK["mass0"], [(_EK["dist"], _EK["base"], 0.85)],
                        model=model)
    VALS["ek_litterature"] = {
        "mach": 0.85, "seats": seats,
        "fuel_t":         r085["fuel"] / 1000.0,
        "time_h":         r085["time"] / 3600.0,
        "t_per_h":        (r085["fuel"] / 1000.0) / (r085["time"] / 3600.0),
        "kg_per_km":      r085["fuel"] / (_EK["dist"] / 1000.0),
        "kg_per_seat_km": r085["fuel"] / (_EK["dist"] / 1000.0) / seats,
        "fuel_15000km_t": r085["fuel"] / (_EK["dist"] / 1000.0) * 15_000.0 / 1000.0,
    }

    # Coût énergétique d'un step-climb (ordre de grandeur, montée non modélisée) :
    # Δm ≈ m·g·Δh / (η·PCI), rendement global η ≈ 0.30, PCI kérosène 43.1 MJ/kg,
    # à la masse de mi-vol (masse initiale − moitié du carburant du vol direct).
    eta, pci = 0.30, 43.1e6
    dh = mod_traj.STEP_CLIMB_FT * FT
    m_mid = _EK["mass0"] - 0.5 * VALS["ek_econ_adaptatif"]["0"]["fixe_fuel_t"] * 1000.0
    VALS["step_cost"] = {
        "eta": eta, "pci_mj_kg": pci / 1e6, "dh_m": dh,
        "m_mid_t": m_mid / 1000.0,
        "dm_kg": m_mid * mod_atm.G * dh / (eta * pci),
    }


def extra_anim(model):
    """Profils EK215 échantillonnés pour l'animation « course des deux
    profils » de l'app (page Présentation) : distance, altitude et carburant
    cumulé à chaque pas d'intégration, pour les trois stratégies. Le CO2 s'en
    déduit exactement (EI constant 3,16 kg/kg) — pas besoin de le stocker."""
    print("extra_anim …  (profils EK215 pour l'animation)")
    cas = mod_traj.compare_step_climbs(MASS_STUDY, DIST_EK, MACH_EK, BASE_EK,
                                       model=model)
    out = {}
    for k in (0, 1, 2):
        c = cas[k]
        if not c["feasible"]:
            continue
        prof = c["result"]["profile"]
        out[str(k)] = {
            "s_km":   [round(p["s"] / 1000.0) for p in prof],
            "fl":     [round(p["alt"] / FT / 100.0) for p in prof],
            "fuel_t": [round((MASS_STUDY - p["mass"]) / 1000.0, 2)
                       for p in prof],
            "time_h": c["time"] / 3600.0,
        }
    VALS["ek_anim"] = out


def main():
    print("Construction du modèle aérodynamique …")
    model = mod_aero.build_aero_model()
    fig_atm()
    fig_geometry()
    fig_ek_route()
    fig_aero_surfaces(model)
    fig_polaire(model)
    fig_ek_moteur(model)
    fig_ei_lnln(model)
    fig_sr_and_cost(model)
    fig_sr_masses(model)
    fig_ek215(model)
    fig_sr_altitudes(model)
    fig_thrust_altitude(model)
    fig_trim_analyse(model)
    fig_cg(model)
    fig_ek_sensibilites(model)
    fig_ek_wind(model)
    fig_econ_ci(model)
    fig_ci_temps(model)
    fig_ek_emissions()
    fig_ek_emissions_sweeps(model)
    extra_values(model)
    extra_revue(model)
    extra_anim(model)
    with open(os.path.join(OUT, "valeurs.json"), "w") as f:
        json.dump(VALS, f, indent=2, ensure_ascii=False)
    print("  → valeurs.json")
    print("Terminé.")


if __name__ == "__main__":
    main()
