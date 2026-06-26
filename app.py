"""
Application Streamlit — Performances Airbus A380 (MGA803)

Interface web interactive regroupant les modules du projet :
atmosphère ISA, conversions de vitesses, aérodynamique (OpenVSP),
propulsion et émissions OACI.

Lancement : streamlit run app.py
"""

import os
import base64
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import atmosphere as mod_atm
import conversion as mod_conv
import aerodynamics as mod_aero
import propulsion as mod_prop
import trim as mod_trim
import performance as mod_perf

KT = 0.514444   # 1 kt en m/s
FT = 0.3048     # 1 ft en m

NAVY = "#1B3A5C"   # accent de l'identité visuelle
RED  = "#D7263D"   # point courant sur les graphes

# Couleur signature par module (palette système Apple) : (foncé, vif)
# foncé = textes/pills sur fond clair ; vif = dégradés de titres, cartes
ACCENTS = {
    "Atmosphère":             ("#0066CC", "#0A84FF"),
    "Conversion":             ("#8944AB", "#BF5AF2"),
    "Aérodynamique":          ("#1D8A3E", "#30D158"),
    "Propulsion & Émissions": ("#C25E00", "#FF9F0A"),
    "Équilibrage (Trim)":     ("#6E6E73", "#8E8E93"),
    "Performance croisière":  ("#54606E", "#8794A4"),
}
# Courbes multi-séries (Mach) — couleurs système Apple
APPLE_SEQ = ["#0A84FF", "#30D158", "#FF9F0A", "#BF5AF2", "#FF375F", "#64D2FF"]

# Graphes : barre d'outils Plotly masquée (rendu présentation, le zoom et
# le survol restent actifs)
PLOTLY_CONF = {"displayModeBar": False}

# Échelle commune des surfaces 3D : parcourt les accents vifs des quatre
# modules (ATM bleu → CONV violet → AERO vert → PROP orange), pour lier
# le relief aux couleurs de l'ensemble du projet plutôt qu'à un dégradé
# monochrome. Partagée par les pages Aérodynamique et Propulsion.
SCALE_MODULES = [
    [0.00, "#0A84FF"],   # ATM  — bleu
    [0.33, "#BF5AF2"],   # CONV — violet
    [0.67, "#30D158"],   # AERO — vert
    [1.00, "#FF9F0A"],   # PROP — orange
]

# Quadrillage dense sur les surfaces 3D (façon mesh MATLAB) : ~n lignes par axe
def _contours(x_vals, y_vals, n=22):
    def axc(vals):
        lo, hi = float(np.min(vals)), float(np.max(vals))
        size = (hi - lo) / n if hi > lo else 1.0
        return dict(show=True, color="rgba(255,255,255,.5)", width=1,
                    start=lo, end=hi, size=size)
    return dict(x=axc(x_vals), y=axc(y_vals))


def _rgba(hexc, a):
    """Couleur rgba à partir d'un hex (#RRGGBB) et d'une opacité."""
    h = hexc.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _mono_scale(acc_v):
    """Échelle de couleurs monochrome (clair → accent) pour les surfaces 3D.
    Bas de l'échelle renforcé (.50) pour éviter le quasi-blanc."""
    return [[0.0, _rgba(acc_v, .50)], [0.5, _rgba(acc_v, .75)],
            [1.0, acc_v]]

# Piles de polices : Helvetica Neue en premier (préférence utilisateur),
# SF Mono pour les chiffres
FONT_UI = ('"Helvetica Neue", Helvetica, -apple-system, BlinkMacSystemFont, '
           'Arial, sans-serif')
FONT_MONO = ('ui-monospace, "SF Mono", SFMono-Regular, Menlo, Monaco, '
             '"Source Code Pro", monospace')

ASSETS = Path(__file__).parent / "assets"
HERO_VIDEO = ASSETS / "video-accueil.mp4"
SIDEBAR_IMG = ASSETS / "sidebar-airflow.jpg"

st.set_page_config(page_title="A380 — MGA803", page_icon="✈️", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Composants UI (maquette module atmosphérique) — CSS scopé aux classes am-*
# uniquement, polices déjà embarquées par Streamlit (Source Sans/Code Pro)
# ---------------------------------------------------------------------------

st.markdown("""
<style>
.am-h1 { font-size: 34px; font-weight: 700; letter-spacing: -.02em;
         color: #1A2230; margin: 0; line-height: 1.2; }
.am-lede { font-size: 15.5px; color: #5B6573; line-height: 1.55;
           max-width: 64ch; margin: 8px 0 0; }
.am-card-title { font-size: 12.5px; font-weight: 700; letter-spacing: .08em;
                 text-transform: uppercase; color: #5B6573; margin: 0 0 6px; }
.am-grid { display: grid; }
.am-grid.cols-1 { grid-template-columns: 1fr; }
.am-grid.cols-2 { grid-template-columns: repeat(2, 1fr); }
.am-grid.cols-3 { grid-template-columns: repeat(3, 1fr); }
.am-grid.cols-4 { grid-template-columns: repeat(4, 1fr); }
.am-grid.cols-5 { grid-template-columns: repeat(5, 1fr); }
.am-metric { padding: 12px 22px 14px 22px; }
.am-grid.cols-2 > .am-metric:nth-child(2n) { border-left: 1px solid #EEF1F5; }
.am-grid.cols-2 > .am-metric:nth-child(n+3) { border-top: 1px solid #EEF1F5; }
.am-grid.cols-3 > .am-metric:not(:nth-child(3n+1)) { border-left: 1px solid #EEF1F5; }
.am-grid.cols-3 > .am-metric:nth-child(n+4) { border-top: 1px solid #EEF1F5; }
.am-grid.cols-4 > .am-metric:not(:nth-child(4n+1)) { border-left: 1px solid #EEF1F5; }
.am-grid.cols-4 > .am-metric:nth-child(n+5) { border-top: 1px solid #EEF1F5; }
.am-grid.cols-5 > .am-metric:not(:nth-child(5n+1)) { border-left: 1px solid #EEF1F5; }
.am-grid.cols-5 > .am-metric:nth-child(n+6) { border-top: 1px solid #EEF1F5; }
.am-mlabel { font-size: 14px; font-weight: 600; color: #1A2230; }
.am-value { font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo,
            "Source Code Pro", monospace;
            font-variant-numeric: tabular-nums; font-size: 42px;
            font-weight: 600; color: #1A2230; letter-spacing: -.01em;
            line-height: 1.15; margin-top: 2px; display: flex;
            align-items: baseline; gap: 8px; }
.am-grid.sm .am-value { font-size: 26px; }
.am-unit { font-size: .45em; font-weight: 600; color: #8B93A1; }
.am-pill { display: inline-block; margin-top: 8px;
           font-family: ui-monospace, "SF Mono", SFMono-Regular, monospace;
           font-size: 13px; font-weight: 600; color: #5B6573;
           background: #F4F6F9; border: 1px solid #E7EBF0;
           border-radius: 99px; padding: 2px 12px; }
.am-ratios { display: grid; background: #F7F9FC; border: 1px solid #E4E8EE;
             border-radius: 10px; padding: 12px 24px; }
.am-ratio { display: flex; align-items: baseline; gap: 10px; }
.am-ratio + .am-ratio { border-left: 1px solid #E4E8EE; padding-left: 24px; }
.am-rlabel { font-size: 13.5px; color: #5B6573; }
.am-math { font-family: "STIX Two Math", "STIX Two Text", "Cambria Math",
           "Times New Roman", serif; font-size: 1.08em; }
.am-math i { font-style: italic; font-family: inherit; }
.am-rvalue { font-family: ui-monospace, "SF Mono", SFMono-Regular, monospace;
             font-size: 19px; font-weight: 600; color: #1A2230; }
/* ---- Accents par module (façon site Apple) ---- */
.am-h1.grad { background: linear-gradient(90deg, #1A2230 30%, var(--acc) 100%);
              -webkit-background-clip: text; background-clip: text;
              -webkit-text-fill-color: transparent; }
.am-wrap .am-card-title { color: var(--acc, #5B6573); }
.am-wrap .am-pill {
    background: color-mix(in srgb, var(--acc, #5B6573) 9%, white);
    border-color: color-mix(in srgb, var(--acc, #5B6573) 22%, white);
    color: var(--acc, #5B6573); }
.am-ratios.tinted {
    background: color-mix(in srgb, var(--acc) 6%, white);
    border-color: color-mix(in srgb, var(--acc) 18%, white); }
.am-ratios.tinted .am-rvalue { color: var(--acc); }
.am-ratios.tinted .am-ratio + .am-ratio {
    border-left-color: color-mix(in srgb, var(--acc) 18%, white); }
/* ---- Titre de module collant en haut au défilement ---- */
[data-testid="stElementContainer"]:has(.am-sticky) {
    position: sticky; top: 0; z-index: 100;
}
/* bandeau de titre : pas de cadre visible, mais un verre dépoli léger qui
   estompe le contenu défilant dessous pour préserver la lisibilité du titre.
   Padding latéral connu sur le conteneur principal, que le bandeau compense
   (marges négatives) pour s'étendre sur toute la largeur de la feuille. */
[data-testid="stMainBlockContainer"], .block-container,
[data-testid="stAppViewBlockContainer"] {
    max-width: 1080px !important;
    margin-left: auto !important; margin-right: auto !important;
    padding-left: 3rem; padding-right: 3rem;
    padding-top: 0rem !important;
    transition: margin .28s cubic-bezier(.4, 0, .2, 1),
                max-width .28s cubic-bezier(.4, 0, .2, 1) !important;
}
/* transition douce de la zone principale (réservation du ruban droit) */
[data-testid="stMain"], section.main {
    transition: padding .28s cubic-bezier(.4, 0, .2, 1); }
.am-head { padding: 8px 3rem; margin-left: -3rem; margin-right: -3rem;
           background: transparent;
           -webkit-backdrop-filter: blur(12px); backdrop-filter: blur(12px); }
.am-head .am-h1 { margin: 0; }
/* ---- Volet gauche : navigation en tuiles d'icônes (toujours ouvert) ---- */
[data-testid="stSidebar"] [role="radiogroup"] { gap: 5px !important; }
[data-testid="stSidebar"] [role="radiogroup"] > label {
    display: flex !important; align-items: center; gap: 0;
    padding: 0 !important; margin: 0 !important; min-height: 50px; cursor: pointer; }
/* cacher le rond radio natif */
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {
    display: none !important; }
/* tuile carrée d'icône (SVG coloré par module injecté en ::before) */
[data-testid="stSidebar"] [role="radiogroup"] > label::before {
    content: ""; order: -1; flex: 0 0 auto;
    width: 42px; height: 42px; border-radius: 12px; margin-right: 13px;
    background-repeat: no-repeat; background-position: center;
    background-size: 21px 21px;
    background-color: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.08);
    transition: background-color .16s, box-shadow .16s; }
[data-testid="stSidebar"] [role="radiogroup"] p {
    margin: 0; font-size: 14.5px; font-weight: 500; color: #B7C6DC;
    white-space: nowrap; transition: color .16s; }
[data-testid="stSidebar"] [role="radiogroup"] > label:hover p { color: #E6EEFB; }
[data-testid="stSidebar"] [role="radiogroup"] > label:hover::before {
    background-color: rgba(255,255,255,.10); }
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {
    color: #fff; font-weight: 600; }
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked)::before {
    background-color: rgba(255,255,255,.13);
    box-shadow: inset 0 0 0 1px rgba(255,255,255,.22), 0 5px 16px -6px rgba(0,0,0,.6); }
/* ---- Sliders façon maquette : pouce blanc à ombre, piste fine arrondie ---- */
[data-testid="stSlider"] div[role="slider"] {
    background-color: #FFFFFF !important;
    border: .5px solid rgba(0, 0, 0, .06) !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, .16), 0 4px 12px rgba(0, 0, 0, .22) !important;
}
/* piste (filled + unfilled) plus épaisse et arrondie — plusieurs cibles DOM */
[data-testid="stSliderTrack"],
[data-testid="stSliderTrack"] > div,
[data-testid="stSlider"] [data-baseweb="slider"] > div > div,
[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div:not([role="slider"]) {
    height: 8px !important;
    border-radius: 999px !important;
}
/* valeur au-dessus du pouce : couleur d'accent du module si défini */
[data-testid="stSliderThumbValue"] { color: var(--acc, #1B3A5C) !important;
    font-variant-numeric: tabular-nums; }
/* ---- Téléphone : grilles resserrées, typo réduite, ratios empilés ---- */
@media (max-width: 640px) {
  .am-h1 { font-size: 26px; }
  .am-lede { font-size: 14px; }
  .am-grid.cols-3, .am-grid.cols-4, .am-grid.cols-5 {
      grid-template-columns: repeat(2, 1fr); }
  .am-grid > .am-metric { border-left: none !important;
      border-top: 1px solid #EEF1F5; padding: 10px 8px 12px; }
  .am-grid:not(.cols-1) > .am-metric:nth-child(2n) {
      border-left: 1px solid #EEF1F5 !important; }
  .am-grid:not(.cols-1) > .am-metric:nth-child(-n+2),
  .am-grid.cols-1 > .am-metric:first-child { border-top: none; }
  .am-value { font-size: 30px; }
  .am-grid.sm .am-value { font-size: 22px; }
  .am-ratios { grid-template-columns: 1fr !important; row-gap: 10px;
      padding: 12px 16px; }
  .am-ratio + .am-ratio { border-left: none; padding-left: 0; }
}
</style>
""", unsafe_allow_html=True)


def fr(x, dec=0):
    """Format numérique : espace pour les milliers (style français)."""
    return f"{x:,.{dec}f}".replace(",", " ")


def page_head(titre, lede="", accent=None):
    # titre seul dans son bloc : son conteneur devient sticky (cf. CSS
    # am-sticky), le lede défile normalement. Bandeau transparent : juste
    # le titre, sans cadre ni image (préférence Kevin).
    cls = "am-h1 grad" if accent else "am-h1"
    style = f' style="--acc:{accent}"' if accent else ""
    st.markdown(f'<div class="am-head am-sticky">'
                f'<h1 class="{cls}"{style}>{titre}</h1></div>',
                unsafe_allow_html=True)
    if lede:
        st.markdown(f'<p class="am-lede">{lede}</p>',
                    unsafe_allow_html=True)


def sym(expr):
    """Expression symbolique en police mathématique (style LaTeX).

    Utiliser <i>…</i> pour les variables et <sub>/<sup> pour les indices :
    sym('<i>θ</i> = <i>T</i>/<i>T</i><sub>0</sub>')
    """
    return f'<span class="am-math">{expr}</span>'


def metric(label, value, unit="", pill=None):
    u = f'<span class="am-unit">{unit}</span>' if unit else ""
    p = f'<div><span class="am-pill">{pill}</span></div>' if pill else ""
    return (f'<div class="am-metric"><div class="am-mlabel">{label}</div>'
            f'<div class="am-value">{value}{u}</div>{p}</div>')


def metrics_card(title, items, cols=2, small=False, accent=None):
    """Grille de métriques façon maquette — à utiliser dans un container."""
    cls = f"am-grid cols-{cols}" + (" sm" if small else "")
    style = f' style="--acc:{accent}"' if accent else ""
    st.markdown(f'<div class="am-wrap"{style}>'
                f'<div class="am-card-title">{title}</div>'
                f'<div class="{cls}">{"".join(items)}</div></div>',
                unsafe_allow_html=True)


def ratios_strip(pairs, accent=None):
    """Bandeau de grandeurs intermédiaires — pairs = [(label, val)]."""
    inner = "".join(f'<div class="am-ratio"><span class="am-rlabel">{l}</span>'
                    f'<span class="am-rvalue">{v}</span></div>'
                    for l, v in pairs)
    cls = "am-ratios tinted" if accent else "am-ratios"
    acc = f"--acc:{accent};" if accent else ""
    st.markdown(f'<div class="{cls}" style="{acc}grid-template-columns:'
                f'repeat({len(pairs)}, 1fr)">{inner}</div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chargement des modèles (mis en cache par Streamlit)
# ---------------------------------------------------------------------------

def _aero_sig():
    """Signature des fichiers de données (mtime) — change si Kevin remplace les
    .history VSPAERO, ce qui invalide automatiquement le cache du modèle aéro."""
    return tuple(os.path.getmtime(f) for f in
                 (mod_aero.DEFAULT_FILE_WB, mod_aero.DEFAULT_FILE_HT))


@st.cache_resource
def _build_aero_cached(sig):
    return mod_aero.build_aero_model()


def load_aero_model():
    return _build_aero_cached(_aero_sig())


@st.cache_data
def aero_curves(delta_it, mach, sig, n_pts=120):
    """Coefficients totaux et WB en fonction de α pour un Mach donné."""
    model = load_aero_model()
    grid = model['f_clwb']
    alphas = np.linspace(grid['x_alpha'][0], grid['x_alpha'][-1], n_pts)
    out = {'alpha': alphas}
    out['cl_t'] = [mod_aero.get_cl_total(model, a, mach, delta_it) for a in alphas]
    out['cd_t'] = [mod_aero.get_cd_total(model, a, mach, delta_it) for a in alphas]
    out['cm_t'] = [mod_aero.get_cm_total(model, a, mach, delta_it) for a in alphas]
    out['cl_wb'] = [mod_aero.get_cl_wb(model, a, mach) for a in alphas]
    out['cd_wb'] = [mod_aero.get_cd_wb(model, a, mach) for a in alphas]
    out['cm_wb'] = [mod_aero.get_cm_wb(model, a, mach) for a in alphas]
    return out


@st.cache_data
def aero_surface(coef, delta_it, sig, n_alpha=45, n_mach=25):
    """Grille (α, M) du coefficient total demandé pour la surface 3D."""
    model = load_aero_model()
    grid = model['f_clwb']
    alphas = np.linspace(grid['x_alpha'][0], grid['x_alpha'][-1], n_alpha)
    machs = np.linspace(grid['y_mach'][0], grid['y_mach'][-1], n_mach)
    # totaux + composantes WB/HT ; WB ignore delta_it, HT le prend
    f = {'CL_t': mod_aero.get_cl_total,
         'CD_t': mod_aero.get_cd_total,
         'CM_t': mod_aero.get_cm_total,
         'CL_wb': lambda mdl, a, m, dit: mod_aero.get_cl_wb(mdl, a, m),
         'CD_wb': lambda mdl, a, m, dit: mod_aero.get_cd_wb(mdl, a, m),
         'CM_wb': lambda mdl, a, m, dit: mod_aero.get_cm_wb(mdl, a, m),
         'CL_ht': mod_aero.get_cl_ht,
         'CD_ht': mod_aero.get_cd_ht,
         'CM_ht': mod_aero.get_cm_ht}[coef]
    z = np.array([[f(model, a, m, delta_it) for m in machs] for a in alphas])
    return alphas, machs, z


@st.cache_data(show_spinner=False)
def perf_cruise(mass, altitude, delta_isa, cost_index, mach_min, mach_max,
                sig, n_pts=41):
    """Balayage du Mach → MRC / LRC / ECON (sig invalide le cache aéro)."""
    model = load_aero_model()
    return mod_perf.cruise_speeds(
        mass, altitude, delta_isa=delta_isa, cost_index=cost_index,
        mach_min=mach_min, mach_max=mach_max, n_pts=n_pts, model=model)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@st.cache_data
def _video_b64(path, mtime):
    """Vidéo encodée en base64 — mtime invalide le cache si le fichier change."""
    return base64.b64encode(Path(path).read_bytes()).decode()


# Rendu dans un iframe (components.html) : le HTML passé à st.markdown est
# inséré via React et Chrome ignore alors l'attribut muted, ce qui bloque
# l'autoplay — dans l'iframe, un script force muted puis play().
_HERO_HTML = """
<style>
body {{ margin: 0; font-family: "Helvetica Neue", Helvetica, -apple-system,
        BlinkMacSystemFont, Arial, sans-serif; }}
.hero {{ position: relative; height: 340px; border-radius: 12px;
         overflow: hidden; }}
.hero video {{ position: absolute; inset: 0; width: 100%; height: 100%;
               object-fit: cover; }}
/* double dégradé bleu nuit : bas → haut + gauche → droite */
.hero .hero-shade {{ position: absolute; inset: 0; background:
  linear-gradient(0deg, rgba(9,21,38,.78) 0%, rgba(9,21,38,.28) 55%,
                  rgba(9,21,38,.05) 100%),
  linear-gradient(90deg, rgba(9,21,38,.55) 0%, rgba(9,21,38,0) 60%); }}
.hero .hero-text {{ position: absolute; left: 36px; bottom: 24px; color: #fff; }}
.hero .overline {{ font-size: 12px; font-weight: 700; letter-spacing: .12em;
                   text-transform: uppercase; opacity: .85;
                   margin-bottom: 6px; }}
.hero .hero-text h1 {{ font-size: 46px; font-weight: 700; margin: 0;
                       letter-spacing: -.02em; line-height: 1.1; }}
.hero .badges {{ display: flex; gap: 8px; margin-top: 14px; }}
.hero .badge {{ font-size: 12.5px; font-weight: 600; padding: 4px 12px;
                border-radius: 99px; background: rgba(255,255,255,.16);
                border: 1px solid rgba(255,255,255,.28);
                -webkit-backdrop-filter: blur(6px);
                backdrop-filter: blur(6px); }}
/* téléphone : titre réduit, badges sur plusieurs lignes */
@media (max-width: 640px) {{
  .hero .hero-text {{ left: 18px; right: 18px; bottom: 16px; }}
  .hero .hero-text h1 {{ font-size: 30px; }}
  .hero .overline {{ font-size: 10.5px; }}
  .hero .badges {{ flex-wrap: wrap; }}
}}
</style>
<div class="hero">
  <video autoplay loop muted playsinline
         src="data:video/mp4;base64,{b64}"></video>
  <div class="hero-shade"></div>
  <div class="hero-text">
    <div class="overline">MGA803 — Analyse et optimisation des performances
    des avions</div>
    <h1>Airbus A380</h1>
    <div class="badges">
      <span class="badge">ÉTS · Été 2026</span>
      <span class="badge">5 modules</span>
      <span class="badge">Trent 970</span>
    </div>
  </div>
</div>
<script>
  const v = document.querySelector('.hero video');
  v.muted = true;
  // Safari peut refuser l'autoplay (réglage par site, économie d'énergie) :
  // dans ce cas, démarrer au premier geste de l'utilisateur.
  const tryPlay = () => v.play().then(
    () => document.removeEventListener('pointerdown', tryPlay),
    () => {{}});
  tryPlay();
  document.addEventListener('pointerdown', tryPlay);
</script>
"""


# Fond global de page
BG_IMG = ASSETS / "background-cockpit.jpg"

# Images de fond des cartes modules de l'accueil (optionnelles : carte
# blanche unie si le fichier manque)
CARD_IMGS = {
    "Atmosphère":             ASSETS / "card-atmosphere.jpg",
    "Conversion":             ASSETS / "card-conversion.jpg",
    "Aérodynamique":          ASSETS / "card-aerodynamique.jpg",
    "Propulsion & Émissions": ASSETS / "card-propulsion.jpg",
    "Équilibrage (Trim)":     ASSETS / "card-trim.jpg",
}

# Pied de page accueil — horizon depuis le cockpit + signature du projet
_FOOTER_HTML = """
<style>
.foot-hero { position: relative; height: 300px; border-radius: 12px;
             overflow: hidden; margin-top: 18px; }
.foot-hero img { position: absolute; inset: 0; width: 100%; height: 100%;
                 object-fit: cover; }
.foot-hero .foot-shade { position: absolute; inset: 0; background:
  linear-gradient(0deg, rgba(9,21,38,.72) 0%, rgba(9,21,38,.15) 45%,
                  rgba(9,21,38,.05) 100%); }
.foot-hero .foot-text { position: absolute; left: 0; right: 0; bottom: 22px;
                        text-align: center; color: #fff; }
.foot-hero .foot-text .foot-authors { font-size: 15px; font-weight: 600; }
.foot-hero .foot-text .foot-course { font-size: 12.5px; opacity: .82;
                                     margin-top: 4px; }
</style>
<div class="foot-hero">
  <img src="data:image/jpeg;base64,{b64}" alt="Horizon depuis le cockpit">
  <div class="foot-shade"></div>
  <div class="foot-text">
    <div class="foot-authors">Rodrigue Fosing · Valentin Durand · Kevin Noah</div>
    <div class="foot-course">MGA803 — Analyse et optimisation des performances
    des avions · ÉTS · Été 2026</div>
  </div>
</div>
"""


_CARDS_CSS = """
<style>
.mod-grid { display: grid; grid-template-columns: repeat(2, 1fr);
            gap: 18px; margin-top: 16px; }
.mod-card { display: block; position: relative; border-radius: 16px;
            padding: 20px 22px; text-decoration: none !important;
            background: rgba(255,255,255,.72);
            -webkit-backdrop-filter: blur(24px) saturate(180%);
            backdrop-filter: blur(24px) saturate(180%);
            border: .5px solid rgba(255,255,255,.7);
            box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 10px 30px rgba(16,24,40,.06);
            transition: transform .18s cubic-bezier(.4,0,.2,1),
                        box-shadow .18s, border-color .18s; }
a.mod-card:hover { transform: translateY(-3px);
    border-color: color-mix(in srgb, var(--mc) 40%, white);
    box-shadow: 0 6px 16px rgba(16,24,40,.05),
                0 20px 44px -14px color-mix(in srgb, var(--mc) 50%, transparent); }
.mod-card .mc-top { display: flex; align-items: center; gap: 12px; }
.mod-card .mc-ico { width: 44px; height: 44px; border-radius: 13px; flex: 0 0 auto;
    display: grid; place-items: center; color: var(--mc);
    background: color-mix(in srgb, var(--mc) 12%, white);
    border: 1px solid color-mix(in srgb, var(--mc) 24%, white); }
.mod-card .mc-ico svg { width: 22px; height: 22px; }
.mod-card .mc-num { font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 13px; font-weight: 700; color: var(--mc); margin-left: auto; }
.mod-card .mc-arrow { color: var(--mc); font-size: 18px; line-height: 1; opacity: .75; }
.mod-card h3 { margin: 15px 0 6px; font-size: 20px; font-weight: 700;
               color: #1A2230; letter-spacing: -.01em; }
.mod-card p { margin: 0; font-size: 14px; color: #5B6573; line-height: 1.5; }
.mod-card.off { opacity: .55; }
.mod-card .mod-pill { display: inline-block; margin-top: 14px;
                      font-size: 11px; font-weight: 700;
                      letter-spacing: .06em; color: #5B6573;
                      background: #F4F6F9; border: 1px solid #E4E8EE;
                      border-radius: 99px; padding: 3px 10px; }
@media (max-width: 900px) { .mod-grid { grid-template-columns: 1fr; } }
@media (max-width: 640px) {
  .mod-card { padding: 18px 18px; }
  .mod-card h3 { font-size: 18px; }
}
</style>
"""


def page_accueil():
    if HERO_VIDEO.exists():
        b64 = _video_b64(str(HERO_VIDEO), HERO_VIDEO.stat().st_mtime)
        st.iframe(_HERO_HTML.format(b64=b64), height=348)
    else:
        st.title("✈️ Performances Airbus A380")
        st.caption("MGA803 — Analyse et optimisation des performances "
                   "des avions · ÉTS, É2026")
    modules = [
        ("01", "Atmosphère", True,
         "Modèle ISA : T, P, ρ, a et ratios θ, δ, σ — de 0 à 20 000 m, "
         "avec écart ΔISA."),
        ("02", "Conversion", True,
         "Conversions TAS / CAS / Mach par les formules isentropiques."),
        ("03", "Aérodynamique", True,
         "Coefficients C<sub>L</sub>, C<sub>D</sub>, C<sub>M</sub> "
         "(données OpenVSP / VSPAERO), downwash, totaux avion."),
        ("04", "Propulsion & Émissions", True,
         "Poussée F<sub>N</sub>, débit carburant W<sub>F</sub> (Trent 970) "
         "et émissions OACI (méthode Boeing Fuel Flow)."),
        ("05", "Équilibrage (Trim)", True,
         "Équilibrage longitudinal : α, δstab et F_N par l'algorithme de "
         "Ghazi & Botez, puis N1 et débit carburant W<sub>F</sub>."),
    ]
    cards = ""
    for num, nom, dispo, desc in modules:
        paths, _ = _NAV_ICONS.get(nom, ("", ""))
        mc = ACCENTS.get(nom, (NAVY, NAVY))[1]
        ico = ('<span class="mc-ico"><svg viewBox="0 0 24 24" fill="none" '
               'stroke="currentColor" stroke-width="1.9" stroke-linecap="round" '
               f'stroke-linejoin="round">{paths}</svg></span>')
        head = (f'<div class="mc-top">{ico}<span class="mc-num">{num}</span>'
                f'<span class="mc-arrow">→</span></div>')
        body = f'{head}<h3>{nom}</h3><p>{desc}</p>'
        if dispo:
            cards += (f'<a class="mod-card" style="--mc:{mc}" target="_self" '
                      f'href="?page={quote(nom)}">{body}</a>')
        else:
            cards += (f'<div class="mod-card off" style="--mc:{mc}">{body}'
                      f'<span class="mod-pill">À VENIR</span></div>')
    st.markdown(_CARDS_CSS + f'<div class="mod-grid">{cards}</div>',
                unsafe_allow_html=True)

    st.caption("Rodrigue Fosing · Valentin Durand · Kevin Noah — "
               "MGA803 · ÉTS · Été 2026")


def _init_slider(key, valeur):
    st.session_state.setdefault(f"{key}_slider", valeur)


def _appliquer_saisie(key, fac, lo, hi):
    val = st.session_state[f"{key}_input"] * fac
    st.session_state[f"{key}_slider"] = min(hi, max(lo, val))


def ruban_saisie(cle):
    """Toggle « Saisie directe » aligné à droite, caché par défaut.

    Retourne (zone principale, zone ruban ou None si replié)."""
    _, c_tog = st.columns([3.4, 1], vertical_alignment="center")
    if c_tog.toggle("Saisie directe", key=cle):
        return st.columns([2.6, 1], gap="medium")
    return st.container(), None


def carte_saisie(champs):
    """Carte du ruban : un number_input par champ, synchronisé au slider
    homonyme (le slider, en unités SI, reste la référence ; le champ est
    redérivé à chaque rendu). champs : liste de
    (label, min_SI, max_SI, step | unites, base_key) où unites est un dict
    {unité: (facteur_vers_SI, pas)} — la première unité est celle par défaut.
    """
    with st.container(border=True, height="stretch"):
        st.markdown('<div class="am-card-title">Saisie directe</div>',
                    unsafe_allow_html=True)
        for label, lo, hi, quatrieme, key in champs:
            unites = quatrieme if isinstance(quatrieme, dict) else None
            if unites:
                st.session_state.setdefault(f"{key}_unite",
                                            next(iter(unites)))
                u = st.session_state[f"{key}_unite"]
                fac, step = unites[u]
                label = f"{label} [{u}]"
            else:
                fac, step = 1.0, quatrieme
            st.session_state[f"{key}_input"] = (
                float(st.session_state[f"{key}_slider"]) / fac)
            st.number_input(label, lo / fac, hi / fac, step=step,
                            key=f"{key}_input",
                            on_change=_appliquer_saisie,
                            args=(key, fac, lo, hi))
            if unites:
                st.radio("Unité", list(unites), key=f"{key}_unite",
                         horizontal=True, label_visibility="collapsed")


# Style « tableau de bord » (porté de la maquette Claude design iPad) —
# classes dash-* confinées, rendues via st.markdown (verre dépoli, indicateurs)
_DASH_CSS = """
<style>
.dash-ind-grid { display:grid; grid-template-columns: repeat(4, 1fr) 1.35fr;
    gap:14px; margin: 4px 0 20px; }
.dash-ind { background: rgba(255,255,255,.72);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    backdrop-filter: blur(24px) saturate(180%);
    border: .5px solid rgba(255,255,255,.75); border-radius:16px;
    box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 10px 30px rgba(16,24,40,.06);
    padding:15px 16px 14px; display:flex; flex-direction:column; gap:6px;
    min-height:92px; justify-content:center; }
.dash-ind .lbl { font-size:11.5px; font-weight:600; color:#8E8E93;
    text-transform:uppercase; letter-spacing:.04em; }
.dash-ind .num { font-family: ui-monospace,"SF Mono",SFMono-Regular,Menlo,monospace;
    font-weight:500; font-size:27px; line-height:1; letter-spacing:-.02em;
    color:#1C1C1E; font-variant-numeric:tabular-nums; }
.dash-ind .num .u { font-size:13px; font-weight:500; color:#8E8E93; margin-left:5px; }
.dash-ind .sym { font-size:12px; color:var(--acc,#0066CC); font-weight:600; }
.dash-ind.ratios { flex-direction:row; padding:12px 6px; gap:0; }
.dash-ratio { flex:1; text-align:center; display:flex; flex-direction:column;
    gap:5px; border-right:.5px solid rgba(60,60,67,.12); }
.dash-ratio:last-child { border-right:none; }
.dash-ratio .g { font-size:16px; color:#6E6E73; font-style:italic;
    font-family:Georgia,"Times New Roman",serif; }
.dash-ratio .rv { font-family: ui-monospace,"SF Mono",monospace; font-size:17px;
    font-weight:500; color:#1C1C1E; font-variant-numeric:tabular-nums; }
.dash-ratio .rl { font-size:9.5px; color:#8E8E93; text-transform:uppercase;
    letter-spacing:.05em; }
.dash-chart-head { display:flex; align-items:flex-start;
    justify-content:space-between; margin:0 2px 6px; gap:12px; }
.dash-chart-head .nm { font-size:16px; font-weight:600; color:#1A2230;
    letter-spacing:-.01em; }
.dash-chart-sub { display:block; font-size:11.5px; color:#8B93A1; margin-top:1px; }
.dash-chart-head .cur { font-family: ui-monospace,"SF Mono",monospace; font-size:13px;
    font-weight:500; color:var(--acc,#0066CC); font-variant-numeric:tabular-nums; }
/* Cartes KPI (maquettes Conversion / Aéro / Prop / Trim) */
.dash-kpi-grid { display:grid; gap:14px; margin:4px 0 20px; }
.dash-kpi { background: rgba(255,255,255,.72);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    backdrop-filter: blur(24px) saturate(180%);
    border:.5px solid rgba(255,255,255,.75); border-radius:16px;
    box-shadow:0 1px 2px rgba(16,24,40,.04),0 10px 30px rgba(16,24,40,.06);
    padding:14px 17px 13px; display:flex; flex-direction:column; gap:5px; }
.dash-kpi .tag { align-self:flex-start; font-size:9.5px; font-weight:700;
    letter-spacing:.07em; color:var(--acc,#0066CC);
    background: color-mix(in srgb, var(--acc,#0066CC) 13%, white);
    padding:2px 7px; border-radius:999px; text-transform:uppercase; }
.dash-kpi .lab { font-size:12.5px; font-weight:600; color:#3A3A3C; }
.dash-kpi .num { font-family: ui-monospace,"SF Mono",monospace; font-weight:500;
    font-size:30px; line-height:1.05; letter-spacing:-.02em; color:#1C1C1E;
    font-variant-numeric:tabular-nums; }
.dash-kpi .num .u { font-size:13px; font-weight:500; color:#8E8E93; margin-left:5px; }
.dash-kpi .desc { font-size:11px; color:#8E8E93; line-height:1.35; }
.dash-kpi.hl { box-shadow:0 0 0 2px var(--acc,#0066CC),
    0 10px 30px rgba(16,24,40,.08); }
/* Grille de détails */
.dash-detgrid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px 22px; }
.dash-dcell { display:flex; flex-direction:column; gap:3px; }
.dash-dcell .dl { font-size:11px; color:#8E8E93; font-weight:600;
    text-transform:uppercase; letter-spacing:.03em; }
.dash-dcell .dv { font-family: ui-monospace,"SF Mono",monospace; font-size:15px;
    font-weight:500; color:#1C1C1E; font-variant-numeric:tabular-nums; }
.dash-dcell .dv .u { font-size:11px; color:#8E8E93; margin-left:3px; }
@media (max-width:920px){ .dash-ind-grid{ grid-template-columns:repeat(2,1fr);}
    .dash-detgrid{ grid-template-columns:repeat(2,1fr);} }
</style>
"""


def _dash_kpi(lab, num, unit="", desc="", tag="", hl=False, acc="#0066CC"):
    """Carte KPI façon maquette : tag optionnel, label, gros chiffre mono, description."""
    t = f'<span class="tag">{tag}</span>' if tag else ""
    u = f'<span class="u">{unit}</span>' if unit else ""
    d = f'<span class="desc">{desc}</span>' if desc else ""
    cls = "dash-kpi hl" if hl else "dash-kpi"
    return (f'<div class="{cls}" style="--acc:{acc}">{t}'
            f'<span class="lab">{lab}</span>'
            f'<span class="num">{num}{u}</span>{d}</div>')


def _dash_dcell(label, value, unit=""):
    """Cellule de détail : label discret + valeur mono."""
    u = f'<span class="u">{unit}</span>' if unit else ""
    return (f'<div class="dash-dcell"><span class="dl">{label}</span>'
            f'<span class="dv">{value}{u}</span></div>')


def _scene3d(xt, yt, zt):
    """Scène 3D avec quadrillage et panneaux visibles (rendu lisible)."""
    def ax(t):
        return dict(title=t, showgrid=True, gridcolor="rgba(60,60,67,.22)",
                    gridwidth=1, showbackground=True,
                    backgroundcolor="rgba(247,249,252,.55)",
                    zerolinecolor="rgba(60,60,67,.35)",
                    showline=True, linecolor="rgba(60,60,67,.3)")
    return dict(xaxis=ax(xt), yaxis=ax(yt), zaxis=ax(zt))


def _dash_ind(lbl, num, unit, sym_txt=""):
    """Carte indicateur (verre dépoli) façon maquette : label, gros chiffre, sous-texte."""
    u = f'<span class="u">{unit}</span>' if unit else ""
    s = f'<span class="sym">{sym_txt}</span>' if sym_txt else ""
    return (f'<div class="dash-ind"><span class="lbl">{lbl}</span>'
            f'<span class="num">{num}{u}</span>{s}</div>')


def page_atm():
    acc_d, acc_v = ACCENTS["Atmosphère"]
    st.markdown(_DASH_CSS + _RP_CSS, unsafe_allow_html=True)

    with st.container(border=True, key="rp_panel"):
        st.markdown('<div class="rp-head">Paramètres</div>'
                    '<div class="rp-sub">Saisie directe ou ajustement au slider'
                    '</div>', unsafe_allow_html=True)
        h = _rp_ctrl("Altitude", 0.0, 20000.0, 10000.0, 50.0, "atm_h", "m")
        disa = _rp_ctrl("ΔISA", -30.0, 30.0, 0.0, 1.0, "atm_disa", "°C")
        if st.button("Réinitialiser", width="stretch"):
            for _k, _d in (("atm_h", 10000.0), ("atm_disa", 0.0)):
                st.session_state[f"{_k}_slider"] = _d
            st.rerun()

    page_head("Atmosphère standard internationale",
              "Conditions ambiantes le long du profil de vol de l'A380 — "
              "modèle ISA · paramètres dans le panneau de droite →", accent=acc_v)

    props = mod_atm.atmosphere(h, disa)

    # ── Bande d'indicateurs (4 cartes + ratios) ────────────────────────────
    ratios = (
        '<div class="dash-ind ratios">'
        f'<div class="dash-ratio"><span class="g">θ</span>'
        f'<span class="rv">{props["theta"]:.4f}</span>'
        f'<span class="rl">T / T₀</span></div>'
        f'<div class="dash-ratio"><span class="g">δ</span>'
        f'<span class="rv">{props["delta"]:.4f}</span>'
        f'<span class="rl">P / P₀</span></div>'
        f'<div class="dash-ratio"><span class="g">σ</span>'
        f'<span class="rv">{props["sigma"]:.4f}</span>'
        f'<span class="rl">ρ / ρ₀</span></div></div>')
    st.markdown(
        f'<div class="dash-ind-grid" style="--acc:{acc_d}">'
        + _dash_ind("Température", f"{props['T']:.2f}", "K",
                    f"{props['T'] - 273.15:+.1f} °C")
        + _dash_ind("Pression", fr(props['P']), "Pa",
                    f"{props['P'] / 100:.1f} hPa")
        + _dash_ind("Masse volumique ρ", f"{props['rho']:.4f}", "kg/m³",
                    f"{props['sigma'] * 100:.1f} % de ρ₀")
        + _dash_ind("Vitesse du son a", f"{props['a']:.1f}", "m/s",
                    f"{props['a'] / KT:.0f} kt")
        + ratios + '</div>', unsafe_allow_html=True)

    # ── Grille 2×2 de profils verticaux (cartes verre dépoli) ──────────────
    hs = np.linspace(0.0, 20000.0, 201)
    profils = [
        ("Température — T(z)", "K", mod_atm.temperature(hs, disa), props['T'],
         f"{props['T']:.2f} K"),
        ("Pression — P(z)", "Pa", mod_atm.pressure(hs, disa), props['P'],
         f"{fr(props['P'])} Pa"),
        ("Masse volumique — ρ(z)", "kg/m³", mod_atm.density(hs, disa),
         props['rho'], f"{props['rho']:.4f} kg/m³"),
        ("Vitesse du son — a(z)", "m/s", mod_atm.speed_of_sound(hs, disa),
         props['a'], f"{props['a']:.1f} m/s"),
    ]
    grille = [st.columns(2, gap="medium"), st.columns(2, gap="medium")]
    for i, (nom, unit, vals, cur, cur_txt) in enumerate(profils):
        with grille[i // 2][i % 2]:
            with st.container(border=True):
                st.markdown(
                    f'<div class="dash-chart-head"><span class="nm">{nom}</span>'
                    f'<span class="cur" style="--acc:{acc_d}">{cur_txt}</span>'
                    f'</div>', unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=vals, y=hs, mode="lines", line=dict(color=acc_v, width=2.6),
                    showlegend=False,
                    hovertemplate=f"{nom} : %{{x:.4g}} {unit}<br>"
                                  "h : %{y:,.0f} m<extra></extra>"))
                fig.add_hline(y=mod_atm.H_TROPO, line_dash="dot", line_width=1.2,
                              line_color="rgba(60,60,67,.30)")
                fig.add_shape(type="line", x0=float(np.min(vals)), x1=cur,
                              y0=h, y1=h,
                              line=dict(color=_rgba(acc_d, .4), width=1, dash="dot"))
                fig.add_trace(go.Scatter(
                    x=[cur], y=[h], mode="markers",
                    marker=dict(color="white", size=11,
                                line=dict(color=acc_d, width=3)),
                    showlegend=False, hoverinfo="skip"))
                fig.update_xaxes(showgrid=True, gridcolor="rgba(60,60,67,.07)",
                                 zeroline=False, color="#8B93A1")
                fig.update_yaxes(title_text="h [m]", range=[0, 20000],
                                 showgrid=False, zeroline=False, color="#8B93A1")
                fig.update_layout(height=240, template="plotly_white",
                                  font=dict(family=FONT_UI), showlegend=False,
                                  margin=dict(t=8, b=32, l=10, r=12),
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, config=PLOTLY_CONF)
    st.caption("Ligne pointillée : tropopause (11 000 m) · point cerclé : "
               f"altitude courante {fr(h)} m")

    # ── Détails repliables ─────────────────────────────────────────────────
    with st.expander("Modèle, formules & inversion P → altitude"):
        st.latex(r"T = T_0 + L\,h + \Delta T_{ISA}\quad(L=-6.5\ \mathrm{K/km})"
                 r"\qquad \rho = \frac{P}{R\,T}\qquad a = \sqrt{\gamma\,R\,T}")
        st.latex(r"P = P_0\left(\frac{T - \Delta T_{ISA}}{T_0}\right)"
                 r"^{-g/(R\,L)}\qquad\text{(pression indépendante de }"
                 r"\Delta T_{ISA})")
        st.caption("Inversion : retrouver l'altitude à partir d'une pression.")
        p_query = st.number_input(r"Pression $P$ [Pa]", 5000.0, 110000.0,
                                  float(props['P']), 100.0)
        h_inv = mod_atm.altitude_from_pressure(p_query, disa)
        st.metric("Altitude h", f"{float(h_inv):,.1f} m")
        st.latex(r"\theta = \frac{T}{T_0}\qquad \delta = \frac{P}{P_0}"
                 r"\qquad \sigma = \frac{\rho}{\rho_0}")


def _solve_speeds(typ, val, h, disa):
    """À partir d'une entrée (Mach si typ='Mach', sinon m/s), renvoie
    (TAS, CAS en m/s, Mach)."""
    if typ == "Mach":
        M = val
        tas = mod_conv.mach_to_tas(M, h, disa)
        cas = mod_conv.mach_to_cas(M, h, disa)
    elif typ == "TAS":
        tas = val
        M = mod_conv.tas_to_mach(tas, h, disa)
        cas = mod_conv.tas_to_cas(tas, h, disa)
    else:  # CAS
        cas = val
        M = mod_conv.cas_to_mach(cas, h, disa)
        tas = mod_conv.cas_to_tas(cas, h, disa)
    return float(tas), float(cas), float(M)


def page_conv():
    acc_d, acc_v = ACCENTS["Conversion"]
    st.markdown(_DASH_CSS + _RP_CSS, unsafe_allow_html=True)

    with st.container(border=True, key="rp_panel"):
        st.markdown('<div class="rp-head">Paramètres</div>'
                    '<div class="rp-sub">Type d\'entrée, vitesse et conditions'
                    '</div>', unsafe_allow_html=True)
        typ = st.segmented_control("Type d'entrée", ["TAS", "CAS", "Mach"],
                                   default="CAS", key="conv_type") or "CAS"
        if typ == "Mach":
            val = _rp_ctrl("Entrée — Mach", 0.20, 0.92, 0.80, 0.01, "conv_mach",
                           "", "{:.2f}")
            val_si = val
        else:
            val = _rp_ctrl(f"Entrée — {typ}", 100.0, 400.0, 280.0, 1.0,
                           "conv_spd", "kt")
            val_si = val * KT
        h = _rp_ctrl("Altitude", 0.0, 13100.0, 10000.0, 50.0, "conv_h", "m")
        disa = _rp_ctrl("ΔISA", -30.0, 30.0, 0.0, 1.0, "conv_disa", "°C")
        if st.button("Réinitialiser", width="stretch"):
            for _k, _d in (("conv_mach", 0.80), ("conv_spd", 280.0),
                           ("conv_h", 10000.0), ("conv_disa", 0.0)):
                st.session_state[f"{_k}_slider"] = _d
            st.rerun()

    page_head("Conversion de vitesses",
              "TAS / CAS / Mach en atmosphère ISA · paramètres dans le panneau "
              "de droite →", accent=acc_v)

    tas, cas, M = _solve_speeds(typ, val_si, h, disa)
    atm = mod_atm.atmosphere(h, disa)

    # ── Bande KPI : TAS / CAS / Mach (entrée surlignée) ────────────────────
    st.markdown(
        '<div class="dash-kpi-grid" style="grid-template-columns:repeat(3,1fr)">'
        + _dash_kpi("TAS · Vitesse vraie", f"{tas / KT:.1f}", "kt",
                    "True Air Speed — vitesse réelle dans la masse d'air",
                    tag="entrée" if typ == "TAS" else "",
                    hl=(typ == "TAS"), acc=acc_d)
        + _dash_kpi("CAS · Vitesse calibrée", f"{cas / KT:.1f}", "kt",
                    "Calibrated Air Speed — vitesse lue au badin",
                    tag="entrée" if typ == "CAS" else "",
                    hl=(typ == "CAS"), acc=acc_d)
        + _dash_kpi("Mach", f"{M:.4f}", "",
                    "M = TAS / a — rapport à la vitesse du son locale",
                    tag="entrée" if typ == "Mach" else "",
                    hl=(typ == "Mach"), acc=acc_d)
        + '</div>', unsafe_allow_html=True)

    # ── Grille de graphes : 3 vitesses vs altitude + TAS&CAS vs entrée ─────
    hs = np.linspace(0.0, 13100.0, 160)
    tas_h, cas_h, mach_h = [], [], []
    for hh in hs:
        t, c, m = _solve_speeds(typ, val_si, hh, disa)
        tas_h.append(t / KT); cas_h.append(c / KT); mach_h.append(m)

    def _vchart(col, title, xs, cur_x, cur_txt, xlab):
        with col:
            with st.container(border=True):
                st.markdown(
                    f'<div class="dash-chart-head"><span class="nm">{title}</span>'
                    f'<span class="cur" style="--acc:{acc_d}">{cur_txt}</span>'
                    f'</div>', unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=xs, y=hs, mode="lines", line=dict(color=acc_v, width=2.6),
                    showlegend=False,
                    hovertemplate=f"{xlab} : %{{x:.4g}}<br>"
                                  "h : %{y:,.0f} m<extra></extra>"))
                fig.add_shape(type="line", x0=float(np.min(xs)), x1=cur_x,
                              y0=h, y1=h,
                              line=dict(color=_rgba(acc_d, .4), width=1, dash="dot"))
                fig.add_trace(go.Scatter(
                    x=[cur_x], y=[h], mode="markers",
                    marker=dict(color="white", size=11,
                                line=dict(color=acc_d, width=3)),
                    showlegend=False, hoverinfo="skip"))
                fig.update_xaxes(showgrid=True, gridcolor="rgba(60,60,67,.07)",
                                 zeroline=False, color="#8B93A1")
                fig.update_yaxes(range=[0, 13100], title_text="h [m]",
                                 showgrid=False, zeroline=False, color="#8B93A1")
                fig.update_layout(height=230, template="plotly_white",
                                  font=dict(family=FONT_UI), showlegend=False,
                                  margin=dict(t=8, b=30, l=10, r=12),
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, config=PLOTLY_CONF)

    g1 = st.columns(2, gap="medium")
    _vchart(g1[0], "TAS vs altitude", tas_h, tas / KT, f"{tas / KT:.1f} kt",
            "TAS [kt]")
    _vchart(g1[1], "CAS vs altitude", cas_h, cas / KT, f"{cas / KT:.1f} kt",
            "CAS [kt]")
    g2 = st.columns(2, gap="medium")
    _vchart(g2[0], "Mach vs altitude", mach_h, M, f"{M:.4f}", "Mach")
    with g2[1]:
        with st.container(border=True):
            st.markdown(
                '<div class="dash-chart-head"><span class="nm">TAS &amp; CAS '
                f'vs entrée</span><span class="cur" style="--acc:{acc_d}">'
                f'à {fr(h)} m</span></div>', unsafe_allow_html=True)
            if typ == "Mach":
                in_si = np.linspace(0.20, 0.92, 120); in_x = in_si
                xlab = "Mach entrée"
            else:
                in_si = np.linspace(100.0, 400.0, 120) * KT; in_x = in_si / KT
                xlab = f"{typ} entrée [kt]"
            t_line, c_line = [], []
            for v in in_si:
                t, c, _ = _solve_speeds(typ, v, h, disa)
                t_line.append(t / KT); c_line.append(c / KT)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=in_x, y=t_line, mode="lines", name="TAS",
                line=dict(color=acc_v, width=2.4),
                hovertemplate="entrée %{x:.3g}<br>TAS %{y:.1f} kt<extra></extra>"))
            fig.add_trace(go.Scatter(x=in_x, y=c_line, mode="lines", name="CAS",
                line=dict(color=acc_d, width=2.2, dash="dot"),
                hovertemplate="entrée %{x:.3g}<br>CAS %{y:.1f} kt<extra></extra>"))
            fig.add_trace(go.Scatter(x=[val], y=[tas / KT], mode="markers",
                marker=dict(color="white", size=11,
                            line=dict(color=acc_d, width=3)),
                showlegend=False, hoverinfo="skip"))
            fig.update_xaxes(showgrid=False, zeroline=False, color="#8B93A1")
            fig.update_yaxes(showgrid=True, gridcolor="rgba(60,60,67,.07)",
                             zeroline=False, color="#8B93A1")
            fig.update_layout(height=230, template="plotly_white",
                font=dict(family=FONT_UI), margin=dict(t=8, b=34, l=10, r=12),
                xaxis_title=xlab, yaxis_title="kt",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(size=9), orientation="h", y=1.04, x=0))
            st.plotly_chart(fig, config=PLOTLY_CONF)

    # ── Détails atmosphériques ─────────────────────────────────────────────
    a0 = mod_atm.A0
    eas = tas * np.sqrt(atm['sigma'])
    qc = mod_atm.P0 * ((1 + 0.2 * (cas / a0) ** 2) ** 3.5 - 1)
    with st.container(border=True):
        st.markdown(
            '<div class="dash-detgrid">'
            + _dash_dcell("Température (SAT)", f"{atm['T'] - 273.15:.1f}", "°C")
            + _dash_dcell("Pression statique", f"{atm['P'] / 100:.1f}", "hPa")
            + _dash_dcell("Densité de l'air", f"{atm['rho']:.4f}", "kg/m³")
            + _dash_dcell("Ratio densité σ", f"{atm['sigma']:.4f}")
            + _dash_dcell("Vitesse du son a", f"{atm['a'] / KT:.1f}", "kt")
            + _dash_dcell("EAS · équivalente", f"{eas / KT:.1f}", "kt")
            + _dash_dcell("Pression d'impact qc", f"{qc / 100:.1f}", "hPa")
            + _dash_dcell("Niveau de vol", f"FL{h / FT / 100:.0f}")
            + '</div>', unsafe_allow_html=True)

    with st.expander("Formules de conversion"):
        st.latex(r"V_{TAS} = M\,a_0\sqrt{\theta}")
        st.latex(r"V_{CAS} = a_0\sqrt{5\left\{\left[\delta\left("
                 r"(1+0.2M^2)^{3.5}-1\right)+1\right]^{1/3.5}-1\right\}}")
        st.latex(r"M = \sqrt{5\left\{\left[\frac{1}{\delta}\left(\left["
                 r"1+0.2\left(\frac{V_{CAS}}{a_0}\right)^{2}\right]^{3.5}"
                 r"-1\right)+1\right]^{1/3.5}-1\right\}}")
        st.caption("TAS ↔ CAS : passage obligatoire par le nombre de Mach.")


def page_aero():
    acc_d, acc_v = ACCENTS["Aérodynamique"]
    st.markdown(_DASH_CSS + _RP_CSS, unsafe_allow_html=True)

    model = load_aero_model()
    grid = model['f_clwb']
    a_min, a_max = float(grid['x_alpha'][0]), float(grid['x_alpha'][-1])
    m_min, m_max = float(grid['y_mach'][0]), float(grid['y_mach'][-1])

    with st.container(border=True, key="rp_panel"):
        st.markdown('<div class="rp-head">Paramètres</div>'
                    '<div class="rp-sub">Incidence, Mach et calage stabilisateur'
                    '</div>', unsafe_allow_html=True)
        alpha = _rp_ctrl("Incidence α", a_min, a_max, 2.0, 0.1, "aero_alpha",
                         "°", "{:.0f}")
        mach = _rp_ctrl("Mach", m_min, m_max, float(min(0.80, m_max)), 0.01,
                        "aero_mach", "", "{:.2f}")
        dit = _rp_ctrl("Calage δit", -10.0, 10.0, 0.0, 0.5, "aero_dit", "°",
                       "{:.0f}")
        if st.button("Réinitialiser", width="stretch"):
            for _k, _d in (("aero_alpha", 2.0),
                           ("aero_mach", float(min(0.80, m_max))),
                           ("aero_dit", 0.0)):
                st.session_state[f"{_k}_slider"] = _d
            st.rerun()

    page_head("Aérodynamique",
              "Coefficients VSPAERO · paramètres dans le panneau de droite →",
              accent=acc_v)

    eps = float(mod_aero.f_downwash(alpha))
    rows = [
        ("Aile + fuselage (WB)",
         mod_aero.get_cl_wb(model, alpha, mach),
         mod_aero.get_cd_wb(model, alpha, mach),
         mod_aero.get_cm_wb(model, alpha, mach)),
        ("Empennage (HT)",
         mod_aero.get_cl_ht(model, alpha, mach, dit),
         mod_aero.get_cd_ht(model, alpha, mach, dit),
         mod_aero.get_cm_ht(model, alpha, mach, dit)),
        ("Avion complet (total)",
         mod_aero.get_cl_total(model, alpha, mach, dit),
         mod_aero.get_cd_total(model, alpha, mach, dit),
         mod_aero.get_cm_total(model, alpha, mach, dit)),
    ]
    cl_t, cd_t, cm_t = rows[2][1], rows[2][2], rows[2][3]

    curves = aero_curves(dit, mach, _aero_sig())
    alphas = curves['alpha']
    cl_arr = np.array(curves['cl_t']); cd_arr = np.array(curves['cd_t'])
    with np.errstate(divide="ignore", invalid="ignore"):
        fin = np.where(cd_arr > 0, cl_arr / cd_arr, -np.inf)
    idx = int(np.argmax(fin))
    cl_opt, cd_opt, fin_opt, a_opt = (cl_arr[idx], cd_arr[idx],
                                      fin[idx], alphas[idx])

    # ── Bande KPI : coefficients totaux + finesse max ──────────────────────
    st.markdown(
        '<div class="dash-kpi-grid" style="grid-template-columns:repeat(4,1fr)">'
        + _dash_kpi("C<sub>L</sub> total", f"{cl_t:.4f}", "",
                    "voilure + fuselage + empennage", acc=acc_d)
        + _dash_kpi("C<sub>D</sub> total", f"{cd_t:.5f}", "",
                    "profil + induite + onde", acc=acc_d)
        + _dash_kpi("C<sub>M</sub> total", f"{cm_t:.4f}", "",
                    "moment rapporté à 25 % CMA", acc=acc_d)
        + _dash_kpi("Finesse L/D max", f"{fin_opt:.1f}", "",
                    f"à α = {a_opt:.1f}° · courante {cl_t / cd_t:.1f}",
                    hl=True, acc=acc_d)
        + '</div>', unsafe_allow_html=True)

    # ── Coefficients = f(α) · pleine largeur ───────────────────────────────
    with st.container(border=True):
        st.markdown('<div class="dash-chart-head"><span class="nm">Coefficients '
                    '= f(α) · total vs aile-fuselage</span><span class="cur" '
                    f'style="--acc:{acc_d}">M {mach:.2f} · δit {dit:+.1f}°</span>'
                    '</div>', unsafe_allow_html=True)
        fig = make_subplots(rows=1, cols=3, subplot_titles=("CL", "CD", "CM"))
        for i, key in enumerate(("cl", "cd", "cm"), start=1):
            cname = key.upper()
            fig.add_trace(go.Scatter(x=alphas, y=curves[f'{key}_t'],
                name="total", legendgroup="t", showlegend=(i == 1),
                line=dict(color=acc_v, width=2),
                hovertemplate=f"α %{{x:.2f}}°<br>{cname} total %{{y:.4f}}"
                              "<extra></extra>"), row=1, col=i)
            fig.add_trace(go.Scatter(x=alphas, y=curves[f'{key}_wb'],
                name="WB seul", legendgroup="wb", showlegend=(i == 1),
                line=dict(color="#8B93A1", dash="dash"),
                hovertemplate=f"α %{{x:.2f}}°<br>{cname} WB %{{y:.4f}}"
                              "<extra></extra>"), row=1, col=i)
            cur = {"cl": cl_t, "cd": cd_t, "cm": cm_t}[key]
            fig.add_trace(go.Scatter(x=[alpha], y=[cur], mode="markers",
                marker=dict(color="white", size=10,
                            line=dict(color=acc_d, width=3)),
                showlegend=False, hoverinfo="skip"), row=1, col=i)
        fig.update_xaxes(title_text="α [°]", showgrid=False, zeroline=False,
                         color="#8B93A1")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(60,60,67,.07)",
                         zeroline=False, color="#8B93A1")
        fig.update_layout(height=300, template="plotly_white",
            font=dict(family=FONT_UI), margin=dict(t=30, b=36, l=10, r=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(size=9), orientation="h", y=1.22, x=0))
        st.plotly_chart(fig, config=PLOTLY_CONF)

    # ── Polaire | Surface 3D ───────────────────────────────────────────────
    p1, p2 = st.columns(2, gap="medium")
    with p1:
        with st.container(border=True):
            st.markdown('<div class="dash-chart-head"><span class="nm">Polaire '
                        'CL – CD</span><span class="cur" '
                        f'style="--acc:{acc_d}">finesse max {fin_opt:.1f}</span>'
                        '</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, cd_opt * 1.12], y=[0, cl_opt * 1.12],
                mode="lines", line=dict(color="#8B93A1", dash="dot", width=1),
                showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=curves['cd_t'], y=curves['cl_t'],
                mode="lines", line=dict(color=acc_v, width=2.4), showlegend=False,
                hovertemplate="CD %{x:.5f}<br>CL %{y:.4f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[cd_opt], y=[cl_opt], mode="markers",
                marker=dict(color=acc_v, size=13, symbol="star"),
                showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[cd_t], y=[cl_t], mode="markers",
                marker=dict(color="white", size=11,
                            line=dict(color=acc_d, width=3)),
                showlegend=False, hoverinfo="skip"))
            fig.update_xaxes(showgrid=True, gridcolor="rgba(60,60,67,.07)",
                             zeroline=False, color="#8B93A1")
            fig.update_yaxes(showgrid=True, gridcolor="rgba(60,60,67,.07)",
                             zeroline=False, color="#8B93A1")
            fig.update_layout(height=300, template="plotly_white",
                font=dict(family=FONT_UI), margin=dict(t=8, b=34, l=10, r=12),
                xaxis_title="CD", yaxis_title="CL",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, config=PLOTLY_CONF)
    with p2:
        with st.container(border=True):
            coef = st.selectbox("Surface 3D",
                ["CL_t", "CL_wb", "CL_ht", "CD_t", "CD_wb", "CD_ht",
                 "CM_t", "CM_wb", "CM_ht"], label_visibility="collapsed")
            a_s, m_s, z = aero_surface(coef, dit, _aero_sig())
            z_cur = {"CL_t": cl_t, "CD_t": cd_t, "CM_t": cm_t,
                     "CL_wb": rows[0][1], "CD_wb": rows[0][2], "CM_wb": rows[0][3],
                     "CL_ht": rows[1][1], "CD_ht": rows[1][2],
                     "CM_ht": rows[1][3]}[coef]
            fig = go.Figure(go.Surface(x=m_s, y=a_s, z=z,
                colorscale=_mono_scale(acc_v), showscale=False,
                contours=_contours(m_s, a_s),
                hovertemplate="Mach %{x:.2f}<br>α %{y:.2f}°<br>"
                              f"{coef} %{{z:.4f}}<extra></extra>"))
            fig.add_trace(go.Scatter3d(x=[mach], y=[alpha], z=[z_cur],
                mode="markers", marker=dict(color="white", size=5,
                line=dict(color=acc_d, width=2)),
                showlegend=False, name="point"))
            fig.update_layout(height=300, template="plotly_white",
                font=dict(family=FONT_UI), margin=dict(t=8, b=8, l=8, r=8),
                scene=_scene3d("Mach", "α", coef))
            st.plotly_chart(fig, config=PLOTLY_CONF)

    # ── Décomposition aile-fuselage / empennage ────────────────────────────
    with st.container(border=True):
        st.markdown(
            '<div class="dash-detgrid">'
            + _dash_dcell("Downwash ε", f"{eps:.2f}", "°")
            + _dash_dcell("Incidence empennage α_ht", f"{alpha - eps + dit:.2f}", "°")
            + _dash_dcell("CL aile-fuselage", f"{rows[0][1]:.4f}")
            + _dash_dcell("CL empennage", f"{rows[1][1]:.4f}")
            + _dash_dcell("CD aile-fuselage", f"{rows[0][2]:.5f}")
            + _dash_dcell("CD empennage", f"{rows[1][2]:.5f}")
            + _dash_dcell("CM aile-fuselage", f"{rows[0][3]:.4f}")
            + _dash_dcell("CM empennage", f"{rows[1][3]:.4f}")
            + '</div>', unsafe_allow_html=True)

    with st.expander("Formules du modèle"):
        st.latex(r"\varepsilon = \varepsilon_0 + \varepsilon_\alpha\,\alpha"
                 r" = 1.18 + 0.37\,\alpha\qquad"
                 r"\alpha_{ht} = \alpha - \varepsilon + \delta_{it}")
        st.latex(r"C_{L,t} = C_{L,wb} + \frac{S_{ht}}{S_{wb}}"
                 r"\left(C_{L,ht}\cos\varepsilon - C_{D,ht}\sin\varepsilon"
                 r"\right)")
        st.latex(r"C_{D,t} = C_{D,wb} + \frac{S_{ht}}{S_{wb}}"
                 r"\left(C_{D,ht}\cos\varepsilon + C_{L,ht}\sin\varepsilon"
                 r"\right)")
        st.latex(r"C_{M,t} = C_{M,wb}"
                 r" + \frac{S_{ht}\,\bar c_{ht}}{S_{wb}\,\bar c_{wb}}\,"
                 r"C_{M,ht}"
                 r" - \frac{S_{ht}\,\bar x_{ht}}{S_{wb}\,\bar c_{wb}}"
                 r"\left(C_{L,ht}\cos\varepsilon - C_{D,ht}\sin\varepsilon"
                 r"\right)"
                 r" + \frac{S_{ht}\,\bar z_{ht}}{S_{wb}\,\bar c_{wb}}"
                 r"\left(C_{L,ht}\sin\varepsilon + C_{D,ht}\cos\varepsilon"
                 r"\right)")
        st.latex(r"\bar x_{ht} = x_{ht}\cos\alpha + z_{ht}\sin\alpha\qquad"
                 r"\bar z_{ht} = z_{ht}\cos\alpha - x_{ht}\sin\alpha")


# Panneau de droite (paramètres de vol) — fixé à droite via :has(.rp-anchor),
# sans réindenter le corps des pages. Le contenu principal laisse la place.
_RP_CSS = """
<style>
.st-key-rp_panel {
    position: fixed !important; top: 0; right: 0; height: 100vh !important;
    width: 330px !important; z-index: 70 !important;
    border: none !important; border-left: 1px solid rgba(60,60,67,.12) !important;
    border-radius: 0 !important;
    background: rgba(255,255,255,.85) !important;
    -webkit-backdrop-filter: blur(26px) saturate(180%);
    backdrop-filter: blur(26px) saturate(180%);
    padding: 1.5rem 1.4rem !important; overflow-y: auto !important;
    box-shadow: -12px 0 44px -24px rgba(16,24,40,.3) !important; }
/* réserve la place du ruban droit au niveau de la zone principale (fiable) ;
   le contenu remplit la largeur restante */
[data-testid="stMain"], section.main { padding-right: 350px !important; }
[data-testid="stMainBlockContainer"] { max-width: none !important; }
/* cartes : autorisées à rétrécir dans la largeur restante (évite le débordement) */
.dash-kpi, .dash-ind, .dash-dcell, .dash-ratio { min-width: 0 !important; }
.dash-kpi .num, .dash-ind .num { overflow: hidden; text-overflow: ellipsis; }
/* graphes Plotly : suivre la largeur du conteneur */
[data-testid="stPlotlyChart"] { width: 100% !important; }
[data-testid="stPlotlyChart"] > div, [data-testid="stPlotlyChart"] .js-plotly-plot {
    width: 100% !important; }
.rp-head { font-size: 16.5px; font-weight: 600; color: #1A2230; letter-spacing: -.01em; }
.rp-sub { font-size: 12.5px; color: #8B93A1; margin: 2px 0 6px; }
.rp-row { display: flex; justify-content: space-between; align-items: baseline;
    margin: 14px 0 2px; }
.rp-row .rp-l { font-size: 13px; font-weight: 600; color: #1A2230; }
.rp-row .rp-r { font-size: 10.5px; color: #8B93A1;
    font-family: ui-monospace, "SF Mono", monospace; }
</style>
"""


def _rp_ctrl(label, lo, hi, default, step, key, unit="", fmt="{:.0f}"):
    """Ligne du panneau droit : libellé + plage, champ ± et slider synchronisés.
    Le slider est la référence ; le champ est redérivé à chaque rendu."""
    _init_slider(key, default)
    rng = (fmt.format(lo) + " — " + fmt.format(hi)
           + (" " + unit if unit else "")).strip()
    st.markdown(f'<div class="rp-row"><span class="rp-l">{label}</span>'
                f'<span class="rp-r">{rng}</span></div>', unsafe_allow_html=True)
    st.session_state[f"{key}_input"] = float(st.session_state[f"{key}_slider"])
    st.number_input(label, float(lo), float(hi), step=float(step),
                    key=f"{key}_input", on_change=_appliquer_saisie,
                    args=(key, 1.0, float(lo), float(hi)),
                    label_visibility="collapsed")
    st.slider(label, float(lo), float(hi), step=float(step),
              key=f"{key}_slider", label_visibility="collapsed")
    return float(st.session_state[f"{key}_slider"])


def page_prop():
    acc_d, acc_v = ACCENTS["Propulsion & Émissions"]
    st.markdown(_DASH_CSS + _RP_CSS, unsafe_allow_html=True)

    # ── Ruban de droite : paramètres de vol (fixé à droite via .st-key-rp_panel)
    with st.container(border=True, key="rp_panel"):
        st.markdown('<div class="rp-head">Paramètres de vol</div>'
                    '<div class="rp-sub">Saisie directe ou ajustement au slider'
                    '</div>', unsafe_allow_html=True)
        n1 = _rp_ctrl("N1", 18.0, 100.0, 90.0, 0.5, "prop_n1", "%", "{:.1f}")
        mach = _rp_ctrl("Mach", 0.0, 0.89, 0.80, 0.01, "prop_mach", "", "{:.2f}")
        h = _rp_ctrl("Altitude", 0.0, 13000.0, 10000.0, 50.0, "prop_h", "m")
        disa = _rp_ctrl("ΔISA", -25.0, 35.0, 0.0, 1.0, "prop_disa", "°C")
        if st.button("Réinitialiser", use_container_width=True):
            for _k, _d in (("prop_n1", 90.0), ("prop_mach", 0.80),
                           ("prop_h", 10000.0), ("prop_disa", 0.0)):
                st.session_state[f"{_k}_slider"] = _d
            st.rerun()

    page_head("Propulsion & émissions — Trent 970",
              "Rolls-Royce Trent 970 · saisie des paramètres dans le panneau "
              "de droite →", accent=acc_v)

    fn = float(mod_prop.get_thrust(n1, mach, h, disa))
    ei = mod_prop.get_emission_indices(n1, mach, h, disa)
    wf = float(ei['WF'])
    theta = float(mod_atm.theta(h, disa))
    delta = float(mod_atm.delta(h, disa))

    # ── Bande d'indicateurs clés (glass, toujours visible) ─────────────────
    st.markdown(
        '<div class="dash-kpi-grid" style="grid-template-columns:repeat(5,1fr)">'
        + _dash_kpi("Poussée F<sub>N</sub> ×4", f"{4 * fn:.3f}", "",
                    f"1 moteur : {fn:.3f}", acc=acc_d)
        + _dash_kpi("Débit W<sub>F</sub>", f"{wf:.4f}", "kg/s",
                    f"{fr(wf * 3600)} kg/h · total 4 moteurs", acc=acc_d)
        + _dash_kpi("EI NOx", f"{ei['EI_NOx']:.2f}", "g/kg",
                    "indice OACI · oxydes d'azote", acc=acc_d)
        + _dash_kpi("EI CO", f"{ei['EI_CO']:.2f}", "g/kg",
                    "monoxyde de carbone", acc=acc_d)
        + _dash_kpi("EI CO₂", f"{ei['EI_CO2']:.2f}", "kg/kg",
                    "dioxyde de carbone", acc=acc_d)
        + '</div>', unsafe_allow_html=True)

    # ── Graphes : famille de Mach monochrome + courbe courante (style maquette)
    n1s = np.linspace(60.0, 100.0, 81)
    machs = np.linspace(0.0, 0.89, 6)
    SUB = "par moteur · famille de Mach 0 → 0,89"

    def _mono2d(col, title, sub, gety, ycur, cur_txt):
        with col:
            with st.container(border=True):
                st.markdown(f'<div class="dash-chart-head"><div>'
                            f'<span class="nm">{title}</span>'
                            f'<span class="dash-chart-sub">{sub}</span></div>'
                            f'<span class="cur" style="--acc:{acc_d}">{cur_txt}</span>'
                            f'</div>', unsafe_allow_html=True)
                fig = go.Figure()
                for i, M in enumerate(machs):           # famille (accent dégradé)
                    op = 0.13 + 0.30 * (i / (len(machs) - 1))
                    fig.add_trace(go.Scatter(x=n1s, y=gety(n1s, M), mode="lines",
                        line=dict(color=_rgba(acc_v, op), width=1.3),
                        showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=n1s, y=gety(n1s, mach), mode="lines",
                    line=dict(color=acc_d, width=3), fill="tozeroy",
                    fillcolor=_rgba(acc_v, .08), showlegend=False,
                    hovertemplate="N1 %{x:.0f} %<br>%{y:.4g}<extra></extra>"))
                fig.add_shape(type="line", x0=n1, x1=n1, y0=0, y1=ycur,
                    line=dict(color=_rgba(acc_d, .45), width=1, dash="dot"))
                fig.add_trace(go.Scatter(x=[n1], y=[ycur], mode="markers",
                    marker=dict(color="white", size=12,
                                line=dict(color=acc_d, width=3)),
                    showlegend=False, hoverinfo="skip"))
                fig.add_annotation(x=n1, y=ycur, text=cur_txt, showarrow=False,
                    bgcolor=acc_d, borderpad=4, xanchor="left", xshift=11, yshift=1,
                    font=dict(color="white", size=11.5, family=FONT_MONO))
                fig.update_xaxes(title_text="N1 [%]", showgrid=False,
                    zeroline=False, color="#8B93A1")
                fig.update_yaxes(showgrid=True, gridcolor="rgba(60,60,67,.07)",
                    zeroline=False, color="#8B93A1", rangemode="tozero")
                fig.update_layout(height=300, template="plotly_white",
                    font=dict(family=FONT_UI), margin=dict(t=8, b=34, l=8, r=8),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, config=PLOTLY_CONF)

    gL, gR = st.columns(2, gap="large")
    _mono2d(gL, "Poussée F<sub>N</sub> vs N1", SUB,
            lambda x, M: mod_prop.get_thrust(x, M, h, disa), fn, f"{fn:.3f}")
    _mono2d(gR, "Débit W<sub>F</sub> vs N1", SUB,
            lambda x, M: mod_prop.get_fuel_flow(x, M, h, disa), wf,
            f"{wf:.3f} kg/s")

    # surfaces 3D monochromes
    machs_3d = np.linspace(0.0, 0.89, 40)
    N1g, Mg = np.meshgrid(n1s, machs_3d)
    s1, s2 = st.columns(2, gap="large")
    surfs = ((mod_prop.get_thrust(N1g, Mg, h, disa),
              "Surface — F<sub>N</sub> (N1, Mach)",
              "poussée par moteur · altitude courante", fn),
             (mod_prop.get_fuel_flow(N1g, Mg, h, disa),
              "Surface — W<sub>F</sub> (N1, Mach)",
              "débit carburant par moteur · altitude courante", wf))
    for col, (zz, nom, sub, z_cur) in zip((s1, s2), surfs):
        with col:
            with st.container(border=True):
                st.markdown(f'<div class="dash-chart-head"><div>'
                            f'<span class="nm">{nom}</span>'
                            f'<span class="dash-chart-sub">{sub}</span></div></div>',
                            unsafe_allow_html=True)
                fig = go.Figure(go.Surface(x=N1g, y=Mg, z=zz,
                    colorscale=_mono_scale(acc_v), showscale=False,
                    contours=_contours(n1s, machs_3d),
                    hovertemplate="N1 %{x:.0f} %<br>Mach %{y:.2f}<br>"
                                  "%{z:.4g}<extra></extra>"))
                fig.add_trace(go.Scatter3d(x=[n1], y=[mach], z=[z_cur],
                    mode="markers", marker=dict(color="white", size=5,
                    line=dict(color=acc_d, width=2)), showlegend=False))
                fig.update_layout(height=320, template="plotly_white",
                    font=dict(family=FONT_UI), margin=dict(t=6, b=6, l=6, r=6),
                    scene=_scene3d("N1", "Mach", ""))
                st.plotly_chart(fig, config=PLOTLY_CONF)

    # ── Émissions (échelle commune) + carte moteur ────────────────────────
    eL, eR = st.columns(2, gap="large")
    with eL:
        with st.container(border=True):
            st.markdown("<div class=\"dash-chart-head\"><div>"
                        "<span class=\"nm\">Indices d'émission vs N1</span>"
                        "<span class=\"dash-chart-sub\">indice d'émission "
                        "[g/kg carburant] · échelle commune</span></div></div>",
                        unsafe_allow_html=True)
            ec = mod_prop.get_emission_indices(n1s, mach, h, disa)
            fig = go.Figure()
            for key, name, color in (("EI_NOx", "NOx", "#FF9F0A"),
                                     ("EI_CO", "CO", "#0A84FF"),
                                     ("EI_UHC", "UHC", "#30D158")):
                fig.add_trace(go.Scatter(x=n1s, y=ec[key], name=name, mode="lines",
                    line=dict(color=color, width=2.6),
                    hovertemplate="N1 %{x:.0f} %<br>" + name +
                                  " %{y:.2f} g/kg<extra></extra>"))
                fig.add_trace(go.Scatter(x=[n1], y=[ei[key]], mode="markers",
                    marker=dict(color="white", size=11,
                                line=dict(color=color, width=3)),
                    showlegend=False, hoverinfo="skip"))
            fig.update_xaxes(title_text="N1 [%]", showgrid=False, zeroline=False,
                color="#8B93A1")
            fig.update_yaxes(showgrid=True, gridcolor="rgba(60,60,67,.07)",
                zeroline=False, color="#8B93A1", rangemode="tozero")
            fig.update_layout(height=300, template="plotly_white",
                font=dict(family=FONT_UI), margin=dict(t=8, b=34, l=8, r=8),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=1.16, x=1, xanchor="right",
                            font=dict(size=11)))
            st.plotly_chart(fig, config=PLOTLY_CONF)
    with eR:
        with st.container(border=True):
            st.markdown('<div class="dash-chart-head"><div>'
                        '<span class="nm">Rolls-Royce Trent 970</span>'
                        '<span class="dash-chart-sub">turboréacteur à fort taux '
                        'de dilution · A380</span></div></div>',
                        unsafe_allow_html=True)
            _eng = ASSETS / "a380-trent970.jpg"
            if _eng.exists():
                st.image(str(_eng), width="stretch")

    # ── Détails repliables (sous le tableau de bord) ───────────────────────
    with st.expander("Grandeurs intermédiaires & formules"):
        ratios_strip([
            (sym('<i>N</i>1<sub>cor</sub>'), f"{n1 / np.sqrt(theta):.2f} %"),
            (sym('<i>W</i><sub>F,C</sub>'),
             f"{wf / (delta * np.sqrt(theta)):.4f} kg/s"),
            (sym('<i>W</i><sub>F,C</sub><sup>REF</sup>'),
             f"{ei['WF_C_REF']:.4f} kg/s"),
            (f"Humidité {sym('<i>ω</i>')}", f"{ei['omega']:.6f}"),
        ], accent=acc_d)
        st.latex(r"F_N = \bar f_{th}\!\left(\tfrac{N1}{\sqrt{\theta}},\,M"
                 r"\right)\delta\qquad "
                 r"W_F = \bar f_{wf}\!\left(\tfrac{N1}{\sqrt{\theta}},\,M"
                 r"\right)\delta\sqrt{\theta}")
        st.latex(r"EI_{NO_x} = EI_{NO_x,C}^{REF}\sqrt{\delta^{1.02}/"
                 r"\theta^{3.3}}\;e^{-19\,(\omega - 0.00634)}\qquad "
                 r"EI_{CO_2} = 3.16\ \mathrm{kg/kg}")

    with st.expander("Masses de polluants émises"):
        cc1, cc2 = st.columns(2)
        t = cc1.number_input("Durée [s]", 1.0, 36000.0, 3600.0, 60.0)
        n_mot = cc2.radio("Moteurs", [1, 4], horizontal=True)
        em = mod_prop.get_emissions(n1, mach, h, disa, duration=t)
        fuel = em['fuel_burn'] * n_mot
        co2 = em['m_CO2'] * n_mot
        metrics_card(f"Sur {fr(t)} s — {n_mot} moteur(s)", [
            metric("Carburant brûlé", fr(fuel, 1), "kg",
                   f"≈ {fr(fuel / 0.8)} L"),
            metric("NOx", fr(em['m_NOx'] * n_mot / 1000, 2), "kg"),
            metric("UHC", fr(em['m_UHC'] * n_mot / 1000, 3), "kg"),
            metric("CO", fr(em['m_CO'] * n_mot / 1000, 2), "kg"),
            metric("CO₂", fr(co2), "kg", f"{co2 / 1000:.2f} t"),
        ], cols=5, small=True, accent=acc_d)


def page_trim():
    acc_d, acc_v = ACCENTS["Équilibrage (Trim)"]
    st.markdown(_DASH_CSS + _RP_CSS, unsafe_allow_html=True)

    with st.container(border=True, key="rp_panel"):
        st.markdown('<div class="rp-head">Configuration & vol</div>'
                    '<div class="rp-sub">Masse, vol, centrage, pente</div>',
                    unsafe_allow_html=True)
        mass = _rp_ctrl("Masse", 300.0, 575.0, 500.0, 1.0, "trim_mass",
                        "t") * 1000.0
        mach = _rp_ctrl("Mach", 0.50, 0.89, 0.80, 0.01, "trim_mach", "", "{:.2f}")
        h = _rp_ctrl("Altitude", 0.0, 13000.0, 10000.0, 100.0, "trim_h", "m")
        disa = _rp_ctrl("ΔISA", -20.0, 20.0, 0.0, 1.0, "trim_disa", "°C")
        xcg = _rp_ctrl("Centrage x_cg", 0.20, 0.45, 0.40, 0.005, "trim_xcg",
                       "MAC", "{:.2f}")
        gamma = _rp_ctrl("Pente γ", -5.0, 8.0, 0.0, 0.1, "trim_gamma", "°",
                         "{:.0f}")

        # ── Critères de convergence (ε de Ghazi & Botez) ───────────────────
        st.markdown('<div class="rp-head" style="margin-top:.6rem">'
                    'Convergence</div>'
                    '<div class="rp-sub">Tolérances ε de l\'algorithme</div>',
                    unsafe_allow_html=True)
        _EPS_DEG = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        _EPS_FN  = [500.0, 100.0, 50.0, 10.0, 1.0]
        st.markdown('<div class="rp-row"><span class="rp-l">ε α</span>'
                    '<span class="rp-r">deg</span></div>', unsafe_allow_html=True)
        eps_alpha = st.select_slider(
            "ε α", options=_EPS_DEG, value=1e-3, key="trim_eps_alpha",
            format_func=lambda v: f"{v:g}", label_visibility="collapsed")
        st.markdown('<div class="rp-row"><span class="rp-l">ε δstab</span>'
                    '<span class="rp-r">deg</span></div>', unsafe_allow_html=True)
        eps_dstab = st.select_slider(
            "ε δstab", options=_EPS_DEG, value=1e-3, key="trim_eps_dstab",
            format_func=lambda v: f"{v:g}", label_visibility="collapsed")
        st.markdown('<div class="rp-row"><span class="rp-l">ε F_N</span>'
                    '<span class="rp-r">N</span></div>', unsafe_allow_html=True)
        eps_fn = st.select_slider(
            "ε F_N", options=_EPS_FN, value=10.0, key="trim_eps_fn",
            format_func=lambda v: f"{v:g}", label_visibility="collapsed")

        if st.button("Réinitialiser", width="stretch"):
            for _k, _d in (("trim_mass", 500.0), ("trim_mach", 0.80),
                           ("trim_h", 10000.0), ("trim_disa", 0.0),
                           ("trim_xcg", 0.40), ("trim_gamma", 0.0)):
                st.session_state[f"{_k}_slider"] = _d
            st.session_state["trim_eps_alpha"] = 1e-3
            st.session_state["trim_eps_dstab"] = 1e-3
            st.session_state["trim_eps_fn"] = 10.0
            st.rerun()

    page_head("Équilibrage longitudinal",
              "Trim — α, δstab, F_N (Ghazi & Botez) · paramètres dans le "
              "panneau de droite →", accent=acc_v)

    model = load_aero_model()
    try:
        r = mod_trim.trim(mass, mach, h, delta_isa=disa, x_cg=xcg,
                          gamma=gamma, model=model,
                          eps_alpha=eps_alpha, eps_fn=eps_fn,
                          eps_dstab=eps_dstab)
    except ValueError as exc:
        st.warning(f"**Pas d'équilibre trouvé à ce point de vol.**\n\n{exc}\n\n"
                   "L'avion est *limité en poussée*. En haute altitude, "
                   "**augmente le Mach** (≈ 0.85) ; sinon **descends** en "
                   "altitude ou **allège**. Plafond ~FL369 à 500 t, "
                   "~FL344 à 560 t (à M0.85).")
        return
    if not r['converged']:
        st.warning(f"Algorithme non convergé en {r['iterations']} itérations "
                   "— résultats indicatifs.")
    if r['thrust_limited']:
        st.info(f"**Avion limité en poussée à ce point de vol.** L'équilibre "
                f"aérodynamique est trouvé (α, δ<sub>stab</sub>, F_N = "
                f"{r['FN']/1000:.0f} kN), mais la poussée requise "
                f"({r['FN_engine']/1000:.0f} kN/moteur) dépasse la poussée max "
                f"disponible → N1 et W_F non définis. **Descends** en altitude, "
                f"**allège** ou **réduis le Mach** pour rester dans l'enveloppe.",
                icon="⚠️")

    # ── Bande KPI : les 3 inconnues + N1, débit, finesse ───────────────────
    if r['thrust_limited']:
        kpi_n1 = _dash_kpi("Régime N1", "—", "",
                           "limité en poussée", acc=acc_d)
        kpi_wf = _dash_kpi("Débit W<sub>F</sub>", "—", "",
                           "limité en poussée", acc=acc_d)
    else:
        kpi_n1 = _dash_kpi("Régime N1", f"{r['N1']:.1f}", "%",
                           f"convergé en {r['iterations']} it.", acc=acc_d)
        kpi_wf = _dash_kpi("Débit W<sub>F</sub>", fr(r['WF_total_kgh']), "kg/h",
                           "total 4 moteurs", acc=acc_d)
    st.markdown(
        '<div class="dash-kpi-grid" style="grid-template-columns:repeat(5,1fr)">'
        + _dash_kpi("Incidence α", f"{r['alpha']:.2f}", "°",
                    "angle d'incidence géométrique", acc=acc_d)
        + _dash_kpi("Calage δ<sub>stab</sub>", f"{r['dstab']:.2f}", "°",
                    "plan horizontal réglable", acc=acc_d)
        + kpi_n1
        + kpi_wf
        + _dash_kpi("Finesse L/D", f"{r['finesse']:.2f}", "",
                    "C<sub>L</sub> / C<sub>D</sub> au point équilibré", acc=acc_d)
        + '</div>', unsafe_allow_html=True)

    # ── Convergence des inconnues (pleine largeur) ─────────────────────────
    hist = r['history']
    its = [hh['it'] for hh in hist]
    with st.container(border=True):
        st.markdown('<div class="dash-chart-head"><span class="nm">Convergence '
                    'des inconnues</span><span class="cur" '
                    f'style="--acc:{acc_d}">{r["iterations"]} itérations</span>'
                    '</div>', unsafe_allow_html=True)
        figc = make_subplots(rows=1, cols=3,
                             subplot_titles=("α [°]", "δstab [°]", "F_N [kN]"))
        series = [([hh['alpha'] for hh in hist], r['alpha'], 1),
                  ([hh['dstab'] for hh in hist], r['dstab'], 2),
                  ([hh['FN'] / 1000.0 for hh in hist], r['FN'] / 1000.0, 3)]
        for ys, conv, col in series:
            figc.add_trace(go.Scatter(x=its, y=ys, mode="lines+markers",
                line=dict(color=acc_v, width=2), marker=dict(size=6),
                showlegend=False,
                hovertemplate="it %{x:.0f}<br>%{y:.3f}<extra></extra>"),
                row=1, col=col)
            figc.add_hline(y=conv, line=dict(color=_rgba(acc_d, .55), width=1,
                           dash="dot"), row=1, col=col)
        figc.update_xaxes(title_text="Itération", color="#8B93A1")
        figc.update_yaxes(gridcolor="rgba(60,60,67,.07)", color="#8B93A1")
        figc.update_layout(height=290, template="plotly_white",
                           font=dict(family=FONT_UI),
                           margin=dict(t=28, b=36, l=10, r=10),
                           plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(figc, config=PLOTLY_CONF)

    # ── Équilibre portance | moment ────────────────────────────────────────
    grid = model['f_clwb']
    alphas = np.linspace(grid['x_alpha'][0], grid['x_alpha'][-1], 100)
    cl_curve = [mod_aero.get_cl_total(model, a, mach, r['dstab']) for a in alphas]
    cl_req = r['L'] / (r['q'] * mod_trim.S_WB)
    dstabs = np.linspace(mod_trim.DSTAB_MIN, mod_trim.DSTAB_MAX, 100)
    m_curve = [mod_trim.moment_total(model, r['alpha'], mach, d, r['FN'],
                                     r['q'], r['weight'], xcg, gamma) / 1e6
               for d in dstabs]
    e1, e2 = st.columns(2, gap="medium")
    with e1:
        with st.container(border=True):
            st.markdown('<div class="dash-chart-head"><span class="nm">Équilibre '
                        'de portance</span><span class="cur" '
                        f'style="--acc:{acc_d}">CL requis {cl_req:.3f}</span>'
                        '</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=alphas, y=cl_curve, mode="lines",
                line=dict(color=acc_v, width=2.4), showlegend=False,
                hovertemplate="α %{x:.2f}°<br>CL %{y:.4f}<extra></extra>"))
            fig.add_hline(y=cl_req, line=dict(color="#8B93A1", width=1, dash="dash"))
            fig.add_trace(go.Scatter(x=[r['alpha']], y=[cl_req], mode="markers",
                marker=dict(color="white", size=11,
                            line=dict(color=acc_d, width=3)),
                showlegend=False, hoverinfo="skip"))
            fig.update_xaxes(showgrid=False, zeroline=False, color="#8B93A1")
            fig.update_yaxes(showgrid=True, gridcolor="rgba(60,60,67,.07)",
                             zeroline=False, color="#8B93A1")
            fig.update_layout(height=270, template="plotly_white",
                font=dict(family=FONT_UI), margin=dict(t=8, b=34, l=10, r=12),
                xaxis_title="α [°]", yaxis_title="CL",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, config=PLOTLY_CONF)
    with e2:
        with st.container(border=True):
            st.markdown('<div class="dash-chart-head"><span class="nm">Équilibre '
                        'de moment</span><span class="cur" '
                        f'style="--acc:{acc_d}">δstab {r["dstab"]:.2f}°</span>'
                        '</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dstabs, y=m_curve, mode="lines",
                line=dict(color=acc_v, width=2.4), showlegend=False,
                hovertemplate="δstab %{x:.2f}°<br>M %{y:.3f} MN·m<extra></extra>"))
            fig.add_hline(y=0.0, line=dict(color="#8B93A1", width=1, dash="dash"))
            fig.add_trace(go.Scatter(x=[r['dstab']], y=[0.0], mode="markers",
                marker=dict(color="white", size=11,
                            line=dict(color=acc_d, width=3)),
                showlegend=False, hoverinfo="skip"))
            fig.update_xaxes(showgrid=False, zeroline=False, color="#8B93A1")
            fig.update_yaxes(showgrid=True, gridcolor="rgba(60,60,67,.07)",
                             zeroline=False, color="#8B93A1")
            fig.update_layout(height=270, template="plotly_white",
                font=dict(family=FONT_UI), margin=dict(t=8, b=34, l=10, r=12),
                xaxis_title="δstab [°]", yaxis_title="M [MN·m]",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, config=PLOTLY_CONF)

    # ── Système d'équilibre : 3 équations + résidus ────────────────────────
    ar = np.radians(r['alpha'] + mod_trim.PHI_T)
    res_v = r['FN'] * np.sin(ar) + r['L'] - r['weight']
    res_h = r['FN'] * np.cos(ar) - r['D']
    res_m = mod_trim.moment_total(model, r['alpha'], mach, r['dstab'], r['FN'],
                                  r['q'], r['weight'], xcg, gamma)
    with st.container(border=True):
        st.markdown(
            '<div class="dash-detgrid" style="grid-template-columns:1fr">'
            + _dash_dcell("Portance · F_N·sin(α+φ_T) + L − mg₀ = 0",
                          f"résidu {res_v:+.1f}", "N")
            + _dash_dcell("Traînée · F_N·cos(α+φ_T) − D = 0",
                          f"résidu {res_h:+.1f}", "N")
            + _dash_dcell("Moment · M_aéro(α,δstab) + M_moteur(F_N) = 0",
                          f"résidu {res_m:+.0f}", "N·m")
            + '</div>', unsafe_allow_html=True)

    # ── Détail des itérations (ligne d'équilibre surlignée) ────────────────
    with st.container(border=True):
        st.markdown("**Détail des itérations**")
        # Formatage en CHAÎNES uniformes : la ligne it.0 (estimé initial) a
        # CL/CD/écarts = None → rendus « — ». Mélanger str (« — ») et float dans
        # une même colonne casse la sérialisation Arrow de st.dataframe
        # (ArrowTypeError « column CL ») → on formate tout en str (homogène).
        def _f(v, n):   return "—" if v is None else f"{v:.{n}f}"
        def _e(v):      return "—" if v is None else f"{v:.2e}"
        df_hist = pd.DataFrame([{
            "it":          hh['it'],
            "α [°]":       round(hh['alpha'], 3),
            "δstab [°]":   round(hh['dstab'], 3),
            "F_N [kN]":    round(hh['FN'] / 1000.0, 1),
            "CL":          _f(hh['CL'], 4),
            "CD":          _f(hh['CD'], 5),
            "|Δα| [°]":    _e(hh['d_alpha']),
            "|ΔF_N| [N]":  _e(hh['d_FN']),
            "|Δδstab| [°]": _e(hh['d_dstab']),
        } for hh in hist])
        _last = len(df_hist) - 1

        def _surligne_equilibre(row):
            ok = r['converged'] and row.name == _last
            return ["background-color: #DDF4E0; font-weight: 600" if ok else ""
                    for _ in row]

        st.dataframe(df_hist.style.apply(_surligne_equilibre, axis=1),
                     hide_index=True, width="stretch")
        if r['converged']:
            st.caption("Ligne verte = itération d'équilibre (convergence atteinte).")


_PERF_CTRL_CSS = """
<style>
/* Barre de contrôles horizontale, collante sous le titre (maquette
   « Croisière & Coût ») — remplace le volet de droite sur cette page. */
.st-key-perf_ctrlbar {
    position: sticky; top: 3.4rem; z-index: 30;
    background: rgba(255,255,255,.62) !important;
    -webkit-backdrop-filter: blur(28px) saturate(180%);
    backdrop-filter: blur(28px) saturate(180%);
    border: .5px solid rgba(255,255,255,.7) !important;
    border-radius: 16px !important;
    box-shadow: 0 1px 2px rgba(16,24,40,.04),
                0 10px 30px rgba(16,24,40,.06) !important;
    padding: .65rem 1.2rem .35rem !important; margin-bottom: 1.1rem; }
.st-key-perf_ctrlbar [data-testid="stSlider"] { padding-top: 0; }
.perf-ctrl-l { font-size: 12px; font-weight: 600; color: #3A3A3C;
    letter-spacing: -.003em; margin: 0 0 -.5rem; white-space: nowrap; }
/* graphes Plotly : suivre la largeur de leur colonne (pas de volet droit ici) */
[data-testid="stPlotlyChart"] { width: 100% !important; }
[data-testid="stPlotlyChart"] > div, [data-testid="stPlotlyChart"] .js-plotly-plot {
    width: 100% !important; }
</style>
"""


def page_perf():
    # Maquette « Croisière & Coût » : accent ORANGE local (fidèle à la maquette),
    # PAS de volet de droite — paramètres dans une barre horizontale collante ;
    # le ruban gauche reste déroulé (cf. bloc Navigation, exclusion de la page).
    acc_d, acc_v, RED = "#C2710A", "#FF9F0A", "#FF3B30"
    st.markdown(_DASH_CSS + _PERF_CTRL_CSS, unsafe_allow_html=True)

    page_head("Vitesses de croisière optimales",
              "MRC · LRC · ECON — portée spécifique et coût d'exploitation "
              "balayés en fonction du nombre de Mach", accent=acc_v)

    # ── Barre de contrôles horizontale (6 paramètres, collante) ────────────
    with st.container(border=True, key="perf_ctrlbar"):
        cols = st.columns(6)
        defs = [
            ("Masse [t]",          300.0,  575.0,   450.0,  1.0,  "perf_mass"),
            ("Altitude [m]",         0.0, 13100.0, 10668.0, 50.0, "perf_h"),
            ("ΔISA [°C]",          -20.0,   20.0,     0.0,  1.0,  "perf_disa"),
            ("Cost Index [kg/min]",  0.0,  200.0,    30.0,  5.0,  "perf_ci"),
            ("Mach min",             0.40,   0.75,    0.50, 0.01, "perf_mmin"),
            ("Mach max",             0.78,   0.92,    0.90, 0.01, "perf_mmax"),
        ]
        vals = {}
        for col, (lab, lo, hi, dflt, step, key) in zip(cols, defs):
            with col:
                st.markdown(f'<div class="perf-ctrl-l">{lab}</div>',
                            unsafe_allow_html=True)
                vals[key] = st.slider(lab, lo, hi, dflt, step, key=key,
                                      label_visibility="collapsed")
    mass = vals["perf_mass"] * 1000.0
    h, disa, ci = vals["perf_h"], vals["perf_disa"], vals["perf_ci"]
    mmin, mmax = vals["perf_mmin"], vals["perf_mmax"]

    if mmax <= mmin:
        st.warning("Le Mach max doit être supérieur au Mach min.")
        return

    r = perf_cruise(mass, h, disa, ci, mmin, mmax, _aero_sig())
    mrc, lrc, econ = r['MRC'], r['LRC'], r['ECON']
    if mrc is None:
        st.info("**Avion limité en poussée sur tout l'intervalle de Mach.** "
                "Aucun débit W_F exploitable → MRC/LRC/ECON indéterminés. "
                "**Descends** en altitude, **allège** ou élargis la plage de Mach.",
                icon="⚠️")
        return

    # ── Bande d'indicateurs : MRC · LRC · ECON · SR max ────────────────────
    def _kpi_speed(o, lab, tag, hl=False):
        if o is None:
            return _dash_kpi(lab, "—", "", "indéterminé", tag=tag, acc=acc_d)
        return _dash_kpi(lab, f"{o['mach']:.3f}", "Mach",
                         f"{o['tas_kt']:.0f} kt · L/D {o['finesse']:.1f}",
                         tag=tag, hl=hl, acc=acc_d)

    ci_tag = "≡ MRC" if ci == 0 else f"CI {ci:.0f}"
    st.markdown(
        '<div class="dash-kpi-grid" style="grid-template-columns:repeat(4,1fr)">'
        + _kpi_speed(mrc, "MRC", "portée max")
        + _kpi_speed(lrc, "LRC", "0.99·SR")
        + _kpi_speed(econ, "ECON", ci_tag, hl=True)
        + _dash_kpi("SR maximale", f"{r['sr_max']/1852.0:.3f}", "NM/kg",
                    f"{r['sr_max']:.0f} m/kg · au point MRC", acc=acc_d)
        + '</div>', unsafe_allow_html=True)

    # ── Données de courbe ──────────────────────────────────────────────────
    curve = r['curve']
    machs = np.asarray(curve['mach'])
    sr    = np.asarray(curve['sr'])
    wf    = np.asarray(curve['wf'])
    ld    = np.asarray(curve['finesse'])
    tas_a = np.asarray(curve['tas'])
    wfh   = wf * 3600.0                              # kg/h
    vsr, vwf, vld = np.isfinite(sr), np.isfinite(wf), np.isfinite(ld)
    MMO = 0.89                                       # Mmo A380

    def _common(fig, ytitle):
        if mmin <= MMO <= mmax:
            fig.add_vline(x=MMO, line=dict(color="#C0C6CF", width=1, dash="dash"),
                          annotation_text="M_MO", annotation_position="top",
                          annotation_font=dict(size=10, color="#8B93A1"))
        fig.update_xaxes(showgrid=False, zeroline=False, color="#8B93A1")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(60,60,67,.07)",
                         zeroline=False, color="#8B93A1")
        fig.update_layout(height=300, template="plotly_white",
            font=dict(family=FONT_UI), margin=dict(t=12, b=40, l=10, r=14),
            xaxis_title="Nombre de Mach", yaxis_title=ytitle,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")

    def _marks(fig, conv):
        """3 marqueurs MRC/LRC/ECON (point blanc cerclé + étiquette)."""
        for o, name, c in ((mrc, "MRC", acc_d), (lrc, "LRC", acc_v),
                           (econ, "ECON", RED)):
            if o is None:
                continue
            y = conv(o)
            if y is None or not np.isfinite(y):
                continue
            fig.add_trace(go.Scatter(x=[o['mach']], y=[y], mode="markers+text",
                marker=dict(color="white", size=11, line=dict(color=c, width=3)),
                text=[name], textposition="top center",
                textfont=dict(family=FONT_MONO, size=10, color=c),
                showlegend=False,
                hovertemplate=f"<b>{name}</b><br>M %{{x:.3f}}<extra></extra>"))

    def _chart_head(nm, sub, cur):
        st.markdown(f'<div class="dash-chart-head"><span class="nm">{nm}</span>'
                    f'<span class="dash-chart-sub">{sub}</span>'
                    f'<span class="cur" style="--acc:{acc_d}">{cur}</span></div>',
                    unsafe_allow_html=True)

    row1 = st.columns(2)
    row2 = st.columns(2)

    # ── 1. Portée spécifique SR(M) ─────────────────────────────────────────
    with row1[0], st.container(border=True):
        _chart_head("Portée spécifique", "SR = TAS / W_F",
                    f"max {r['sr_max']:.0f} m/kg")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=machs[vsr], y=sr[vsr], mode="lines",
            line=dict(color=acc_v, width=2.6), fill="tozeroy",
            fillcolor=_rgba(acc_v, .12), showlegend=False,
            hovertemplate="M %{x:.3f}<br>SR %{y:.1f} m/kg<extra></extra>"))
        _marks(fig, lambda o: o['sr'])
        _common(fig, "SR [m/kg]")
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig, config=PLOTLY_CONF)

    # ── 2. Coût d'exploitation : carburant + temps (Cost Index) ────────────
    fuel_nm  = (wf / tas_a) * 1852.0                 # coût carburant [kg/NM]
    time_nm  = (ci / 60.0 / tas_a) * 1852.0          # coût temps     [kg/NM]
    total_nm = np.asarray(curve['cost']) * 1852.0    # coût total     [kg/NM]
    C_FUEL = "#3A3A3C"                                # carburant (encre)
    with row1[1], st.container(border=True):
        _chart_head("Coût d'exploitation", "carburant + temps",
                    f"ECON M{econ['mach']:.3f}" if econ else "—")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=machs[vsr], y=total_nm[vsr], mode="lines",
            name="Coût total", line=dict(color=acc_v, width=2.8),
            hovertemplate="M %{x:.3f}<br>Total %{y:.2f} kg/NM<extra></extra>"))
        fig.add_trace(go.Scatter(x=machs[vsr], y=fuel_nm[vsr], mode="lines",
            name="Carburant", line=dict(color=C_FUEL, width=2.0),
            hovertemplate="M %{x:.3f}<br>Carburant %{y:.2f} kg/NM<extra></extra>"))
        fig.add_trace(go.Scatter(x=machs[vsr], y=time_nm[vsr], mode="lines",
            name="Temps", line=dict(color=RED, width=2.0),
            hovertemplate="M %{x:.3f}<br>Temps %{y:.2f} kg/NM<extra></extra>"))
        if econ is not None:
            ye = (econ['wf'] + ci / 60.0) / econ['tas'] * 1852.0
            fig.add_trace(go.Scatter(x=[econ['mach']], y=[ye], mode="markers+text",
                marker=dict(color="white", size=11, line=dict(color=RED, width=3)),
                text=["ECON"], textposition="top center",
                textfont=dict(family=FONT_MONO, size=10, color=RED),
                showlegend=False, hovertemplate="<b>ECON</b><extra></extra>"))
            fig.add_vline(x=econ['mach'],
                          line=dict(color="#9AA3AF", width=1, dash="dot"))
        _common(fig, "Coût [kg/NM]")
        fig.update_yaxes(rangemode="tozero")
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.0,
            xanchor="right", x=1.0, bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, config=PLOTLY_CONF)

    # ── 3. Finesse L/D(M) ──────────────────────────────────────────────────
    with row2[0], st.container(border=True):
        ld_max = float(np.nanmax(ld)) if np.any(vld) else float("nan")
        _chart_head("Finesse", "L/D = f(Mach)",
                    f"max {ld_max:.1f}" if np.isfinite(ld_max) else "—")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=machs[vld], y=ld[vld], mode="lines",
            line=dict(color=acc_v, width=2.6), showlegend=False,
            hovertemplate="M %{x:.3f}<br>L/D %{y:.2f}<extra></extra>"))
        _marks(fig, lambda o: o['finesse'])
        _common(fig, "Finesse  L/D")
        st.plotly_chart(fig, config=PLOTLY_CONF)

    # ── 4. Débit carburant W_F(M) ──────────────────────────────────────────
    with row2[1], st.container(border=True):
        _chart_head("Débit carburant", "W_F = f(Mach)",
                    f"{econ['wf_kgh']:.0f} kg/h @ ECON" if econ else "—")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=machs[vwf], y=wfh[vwf], mode="lines",
            line=dict(color=acc_v, width=2.6), fill="tozeroy",
            fillcolor=_rgba(acc_v, .12), showlegend=False,
            hovertemplate="M %{x:.3f}<br>W_F %{y:.0f} kg/h<extra></extra>"))
        _marks(fig, lambda o: o['wf_kgh'])
        _common(fig, "W_F [kg/h]")
        st.plotly_chart(fig, config=PLOTLY_CONF)

    if ci == 0:
        st.caption("Cost Index = 0 → coût temps nul ; le coût total se réduit au "
                   "carburant, donc ECON coïncide avec MRC.")

    # ── Récapitulatif & modèle (volet dépliable, ouvert par défaut) ────────
    with st.expander("Vitesses optimales — récapitulatif & modèle", expanded=True):
        lignes = []
        for cle, o in (("MRC", mrc), ("LRC", lrc), ("ECON", econ)):
            if o is None:
                lignes.append({"Régime": cle})
                continue
            lignes.append({
                "Régime":       cle,
                "Mach":         round(o['mach'], 4),
                "TAS [kt]":     round(o['tas_kt'], 1),
                "TAS [m/s]":    round(o['tas'], 1),
                "SR [m/kg]":    round(o['sr'], 1),
                "SR [NM/kg]":   round(o['sr_nm_per_kg'], 4),
                "W_F [kg/h]":   round(o['wf_kgh'], 0),
                "L/D":          round(o['finesse'], 2),
            })
        st.dataframe(pd.DataFrame(lignes), hide_index=True, width="stretch")
        st.markdown(
            '<div style="display:flex;flex-wrap:wrap;gap:8px;margin:.5rem 0 .2rem">'
            + "".join(
                f'<span style="font-family:{FONT_MONO};font-size:12px;'
                'color:#6E6E73;background:rgba(120,120,128,.08);padding:5px 10px;'
                f'border-radius:8px">{t}</span>'
                for t in ("S_wb = 859 m²", "c̄_w = 11 m", "x_cg = 0.40 MAC",
                          "φ_T = 2°", "M_MO = 0.89"))
            + '</div>', unsafe_allow_html=True)
        cap = ("SR = TAS ⁄ W_F (portée par kg de carburant). **MRC** maximise la "
               "portée spécifique ; **LRC** est la vitesse plus rapide donnant "
               "encore 99 % de la SR maximale ; **ECON** minimise le coût total "
               "= carburant + temps, le poids du temps étant fixé par le Cost "
               "Index — coût/NM = (W_F + 60·CI) ⁄ TAS. Aéro near-field (OpenVSP) "
               "+ extrapolation spline au-delà de M0.7, équilibrage Ghazi & Botez.")
        if ci == 0:
            cap += " Cost Index = 0 → ECON coïncide avec MRC."
        st.caption(cap)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

PAGES = {
    "Accueil": page_accueil,
    "Atmosphère": page_atm,
    "Conversion": page_conv,
    "Aérodynamique": page_aero,
    "Propulsion & Émissions": page_prop,
    "Équilibrage (Trim)": page_trim,
    "Performance croisière": page_perf,
}

@st.cache_data
def _img_b64(path, mtime):
    """Image encodée en base64 — mtime invalide le cache si le fichier change."""
    return base64.b64encode(Path(path).read_bytes()).decode()


def apply_page_background():
    """Fond dégradé clair épuré (style iPad), sans image, teinté à l'accent
    du module actif. Cartes en verre dépoli sur fond clair.

    CSS limité à des propriétés de fond — dégradation sans risque si la
    structure DOM change.
    """
    nav = st.session_state.get("nav", "Accueil")
    vif = ACCENTS.get(nav, (NAVY, "#3E6B99"))[1]
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background:
          radial-gradient(120% 80% at 78% -10%, {vif}1A, transparent 55%),
          linear-gradient(180deg, #EEF3FA 0%, #F3F6FA 42%, #EFF1F4 100%);
        background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{ background: transparent; }}
    /* cartes en verre dépoli sur le fond clair */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(255, 255, 255, .72);
        -webkit-backdrop-filter: blur(24px) saturate(180%);
        backdrop-filter: blur(24px) saturate(180%);
        border-radius: 16px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, .04),
                    0 10px 30px rgba(16, 24, 40, .06);
    }}
    </style>
    """, unsafe_allow_html=True)


# Icônes (lucide) + couleur par module pour la nav latérale
_NAV_ICONS = {
    "Accueil": ('<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'
                '<path d="M9 22V12h6v10"/>', "#9FB1CB"),
    "Atmosphère": ('<path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/>',
                   "#5AC8FA"),
    "Conversion": ('<path d="m16 3 4 4-4 4"/><path d="M20 7H4"/>'
                   '<path d="m8 21-4-4 4-4"/><path d="M4 17h16"/>', "#A78BFA"),
    "Aérodynamique": ('<path d="M12.8 19.6A2 2 0 1 0 14 16H2"/>'
                      '<path d="M17.5 8a2.5 2.5 0 1 1 2 4H2"/>'
                      '<path d="M9.8 4.4A2 2 0 1 1 11 8H2"/>', "#4FD0C2"),
    "Propulsion & Émissions": ('<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2'
                               '-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 '
                               '2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 '
                               '1-3a2.5 2.5 0 0 0 2.5 2.5z"/>', "#FF9F0A"),
    "Équilibrage (Trim)": ('<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3'
                           '-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13'
                           '-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/>'
                           '<path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>', "#5BD07A"),
    "Performance croisière": ('<path d="m12 14 4-4"/>'
                              '<path d="M3.34 19a10 10 0 1 1 17.32 0"/>', "#9AA7B6"),
}


def apply_sidebar_background():
    """Ruban latéral façon maquette : dégradé navy, nav à icônes colorées,
    barre d'accent sur l'item actif. CSS confiné à [data-testid="stSidebar"]."""
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background:
          radial-gradient(420px 300px at 24% 6%, rgba(120,165,225,.16), transparent 70%),
          radial-gradient(520px 420px at 80% 102%, rgba(255,159,10,.10), transparent 70%),
          linear-gradient(176deg, #122844 0%, #0C1C33 52%, #091322 100%) !important;
        transition: width .28s cubic-bezier(.4, 0, .2, 1),
                    min-width .28s cubic-bezier(.4, 0, .2, 1) !important;
    }
    [data-testid="stSidebarContent"], [data-testid="stSidebarUserContent"] {
        background: transparent !important; }
    /* marque */
    .sb-brand { display:flex; align-items:center; gap:13px; padding:14px 2px 12px; }
    .sb-badge { width:44px; height:44px; border-radius:13px; display:grid;
        place-items:center; font-family:ui-monospace,"SF Mono",monospace;
        font-size:15px; font-weight:600; letter-spacing:-.02em; color:#DBE6F5;
        flex:0 0 auto;
        background:linear-gradient(160deg, rgba(255,255,255,.16), rgba(255,255,255,.03));
        border:1px solid rgba(255,255,255,.16);
        box-shadow:inset 0 1px 0 rgba(255,255,255,.18), 0 4px 12px rgba(0,0,0,.25); }
    .sb-name { font-size:15px; font-weight:600; letter-spacing:-.01em; color:#fff; }
    .sb-role { font-size:12.5px; color:#8EA2BD; margin-top:1px; }
    .sb-seclabel { font-size:10.5px; font-weight:700; letter-spacing:.16em;
        color:#5D728F; text-transform:uppercase; padding:8px 0 6px 24px; }
    /* items de nav : tuile carrée centrée dans le rail via margin auto */
    .sb-nav { display:flex; flex-direction:column; gap:5px; padding:2px 0; }
    a.sb-item { --c:#93A6C1; position:relative; display:flex; align-items:center;
        gap:0; height:52px; padding:0; color:#B7C6DC;
        text-decoration:none !important; transition:color .16s; }
    a.sb-item .ico { width:44px; height:44px; border-radius:13px; flex:0 0 auto;
        margin:0 auto; display:grid; place-items:center; color:var(--c);
        background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.08);
        transition:margin .26s cubic-bezier(.4,0,.2,1), background .16s,
                   box-shadow .16s, border-color .16s; }
    a.sb-item .ico svg { width:21px; height:21px; }
    a.sb-item .lbl { font-size:14.5px; font-weight:500; letter-spacing:-.01em;
        color:inherit; margin-left:0; }
    a.sb-item:hover { color:#E6EEFB; }
    a.sb-item:hover .ico { background:rgba(255,255,255,.10); }
    a.sb-item.active { color:#fff; }
    a.sb-item.active .ico {
        background:color-mix(in srgb, var(--c) 24%, transparent);
        border-color:color-mix(in srgb, var(--c) 45%, transparent);
        box-shadow:inset 0 0 0 1px color-mix(in srgb, var(--c) 30%, transparent),
                   0 5px 16px -5px var(--c); }
    a.sb-item.active .lbl { font-weight:600; }
    /* pied ÉTS, collé en bas */
    .sb-foot { margin-top:auto; padding:16px 4px 6px;
        border-top:1px solid rgba(255,255,255,.07); }
    .sb-foot .ft { font-size:12px; color:#8A9CB6; line-height:1.45; }
    .sb-foot .fm { display:inline-flex; align-items:center; gap:6px; margin-top:9px;
        font-family:ui-monospace,"SF Mono",monospace; font-size:11px; color:#6F829C;
        padding:4px 9px; border-radius:7px; background:rgba(255,255,255,.05);
        border:1px solid rgba(255,255,255,.07); }
    .sb-foot .fm .pip { width:6px; height:6px; border-radius:50%; background:#5BD07A;
        box-shadow:0 0 7px #5BD07A; }
    /* remplir la hauteur pour coller le pied en bas */
    [data-testid="stSidebarUserContent"] { display:flex; flex-direction:column;
        min-height:calc(100vh - 2rem); }
    </style>
    """, unsafe_allow_html=True)

    # Icônes SVG colorées par module, posées en ::before sur chaque label radio
    rules = []
    for i, name in enumerate(PAGES, start=1):
        paths, color = _NAV_ICONS.get(name, ("", "#9FB1CB"))
        svg = ("<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' "
               "viewBox='0 0 24 24' fill='none' stroke='" + color + "' "
               "stroke-width='1.9' stroke-linecap='round' "
               "stroke-linejoin='round'>" + paths + "</svg>")
        uri = "data:image/svg+xml," + quote(svg)
        rules.append('[data-testid="stSidebar"] [role="radiogroup"] > '
                     'label:nth-child(' + str(i) + ')::before '
                     '{ background-image: url("' + uri + '"); }')
    st.markdown("<style>" + "".join(rules) + "</style>", unsafe_allow_html=True)


# Ruban gauche réduit (icônes seules) dans un module ; déroulé à l'Accueil
_RAIL_CSS = """
<style>
/* rail gauche en flux normal, juste rétréci : Streamlit réserve sa place */
[data-testid="stSidebar"] { width: 86px !important; min-width: 86px !important;
    overflow: hidden !important; }
[data-testid="stSidebar"] [role="radiogroup"] p { display: none !important; }
[data-testid="stSidebar"] [role="radiogroup"] > label { justify-content: center; }
[data-testid="stSidebar"] [role="radiogroup"] > label::before {
    margin-right: 0 !important; }
[data-testid="stSidebar"] .sb-meta,
[data-testid="stSidebar"] .sb-seclabel,
[data-testid="stSidebar"] .sb-foot { display: none !important; }
[data-testid="stSidebar"] .sb-brand { justify-content: center;
    padding-left: 0 !important; padding-right: 0 !important; }
</style>
"""


# Navigation par les cartes de l'accueil : ?page=<module> dans l'URL
if "page" in st.query_params:
    _cible = st.query_params["page"]
    if _cible in PAGES:
        st.session_state.nav = _cible
    st.query_params.clear()

apply_page_background()
apply_sidebar_background()

# Marque + label « Module » (en haut du volet)
st.sidebar.markdown(
    '<div class="sb-brand"><div class="sb-badge">380</div>'
    '<div class="sb-meta"><div class="sb-name">A380 — MGA803</div>'
    '<div class="sb-role">Performances avion</div></div></div>'
    '<div class="sb-seclabel">Module</div>', unsafe_allow_html=True)

# Navigation native (rerun léger, AUCUN rechargement). Les icônes colorées par
# module sont injectées en CSS (::before) dans apply_sidebar_background().
choix_page = st.sidebar.radio(
    "Module", list(PAGES), key="nav", label_visibility="collapsed")

st.sidebar.markdown(
    '<div class="sb-foot"><div class="ft">Analyse et optimisation des '
    'performances des avions</div>'
    '<div class="fm"><span class="pip"></span>ÉTS · É2026</div></div>',
    unsafe_allow_html=True)

# Réduit le ruban gauche dans un module ; reste déroulé à l'Accueil et sur la
# page Performance croisière (maquette « Croisière & Coût » : pas de volet droit,
# contrôles en barre horizontale → le ruban gauche se déroule en grand).
if choix_page not in ("Accueil", "Performance croisière"):
    st.markdown(_RAIL_CSS, unsafe_allow_html=True)

PAGES[choix_page]()
