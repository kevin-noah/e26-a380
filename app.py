"""
Application Streamlit — Performances Airbus A380 (MGA803)

Interface web interactive regroupant les modules du projet :
atmosphère ISA, conversions de vitesses, aérodynamique (OpenVSP),
propulsion et émissions OACI.

Lancement : streamlit run app.py
"""

import base64
from pathlib import Path
from urllib.parse import quote

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import atmosphere as mod_atm
import conversion as mod_conv
import aerodynamics as mod_aero
import propulsion as mod_prop

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
}
# Courbes multi-séries (Mach) — couleurs système Apple
APPLE_SEQ = ["#0A84FF", "#30D158", "#FF9F0A", "#BF5AF2", "#FF375F", "#64D2FF"]

# Graphes : barre d'outils Plotly masquée (rendu présentation, le zoom et
# le survol restent actifs)
PLOTLY_CONF = {"displayModeBar": False}

# Échelles des surfaces 3D, assorties à la couleur du module
SCALE_AERO = [[0.0, "#F2FBF5"], [0.5, "#30D158"], [1.0, "#0B3D1B"]]
SCALE_PROP = [[0.0, "#FFF6EA"], [0.5, "#FF9F0A"], [1.0, "#6B3A00"]]

# Piles de polices : Helvetica Neue en premier (préférence utilisateur),
# SF Mono pour les chiffres
FONT_UI = ('"Helvetica Neue", Helvetica, -apple-system, BlinkMacSystemFont, '
           'Arial, sans-serif')
FONT_MONO = ('ui-monospace, "SF Mono", SFMono-Regular, Menlo, Monaco, '
             '"Source Code Pro", monospace')

ASSETS = Path(__file__).parent / "assets"
HERO_VIDEO = ASSETS / "video-accueil.mp4"
SIDEBAR_IMG = ASSETS / "sidebar-airflow.jpg"

st.set_page_config(page_title="A380 — MGA803", page_icon="✈️", layout="wide")

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
[data-testid="stMainBlockContainer"] {
    padding-left: 3rem; padding-right: 3rem;
}
.am-head { padding: 8px 3rem; margin-left: -3rem; margin-right: -3rem;
           background: transparent;
           -webkit-backdrop-filter: blur(12px); backdrop-filter: blur(12px); }
.am-head .am-h1 { margin: 0; }
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

@st.cache_resource
def load_aero_model():
    return mod_aero.build_aero_model()


@st.cache_data
def aero_curves(delta_it, mach, n_pts=120):
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
def aero_surface(coef, delta_it, n_alpha=45, n_mach=25):
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
            gap: 18px; margin-top: 10px; }
.mod-card { display: block; position: relative; background: #fff;
            border: 1px solid #E4E8EE; border-radius: 12px;
            padding: 22px 26px; text-decoration: none !important;
            transition: border-color .15s, box-shadow .15s, transform .15s; }
a.mod-card:hover { border-color: var(--mc, #1B3A5C);
                   transform: translateY(-2px);
                   box-shadow: 0 6px 18px rgba(16, 24, 40, .08); }
.mod-card .mod-head { display: flex; justify-content: space-between;
                      align-items: center; }
.mod-card .mod-num { font-family: ui-monospace, SFMono-Regular, monospace;
                     font-size: 13px; font-weight: 700;
                     color: var(--mc, #8B93A1); }
.mod-card .mod-arrow { color: var(--mc, #5B6573); font-size: 19px;
                       line-height: 1; }
.mod-card h3 { margin: 10px 0 8px; font-size: 22px; font-weight: 700;
               color: #1A2230; }
.mod-card p { margin: 0; font-size: 14.5px; color: #5B6573;
              line-height: 1.55; }
.mod-card.off { opacity: .55; }
.mod-card .mod-pill { display: inline-block; margin-top: 14px;
                      font-size: 11px; font-weight: 700;
                      letter-spacing: .06em; color: #5B6573;
                      background: #F4F6F9; border: 1px solid #E4E8EE;
                      border-radius: 99px; padding: 3px 10px; }
@media (max-width: 900px) { .mod-grid { grid-template-columns: 1fr; } }
@media (max-width: 640px) {
  .mod-card { padding: 18px 20px; }
  .mod-card h3 { font-size: 19px; }
  .mod-card p { font-size: 13.5px; }
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
        ("05", "Équilibrage (Trim)", False,
         "Équilibrage longitudinal de l'avion."),
    ]
    cards = ""
    for num, nom, dispo, desc in modules:
        head = (f'<div class="mod-head"><span class="mod-num">{num}</span>'
                f'<span class="mod-arrow">→</span></div>')
        body = f'{head}<h3>{nom}</h3><p>{desc}</p>'
        mc = ACCENTS.get(nom, (NAVY, NAVY))[1]
        style = f"--mc:{mc}"
        img = CARD_IMGS.get(nom)
        if img is not None and img.exists():
            b64i = _img_b64(str(img), img.stat().st_mtime)
            # voile blanc dégradé : opaque côté texte, image visible à droite
            style += (";background:"
                      "linear-gradient(115deg, rgba(255,255,255,.97) 0%, "
                      "rgba(255,255,255,.88) 52%, rgba(255,255,255,.45) "
                      "100%), "
                      f"url(data:image/jpeg;base64,{b64i}) center / cover "
                      "no-repeat")
        if dispo:
            cards += (f'<a class="mod-card" style="{style}" '
                      f'target="_self" '
                      f'href="?page={quote(nom)}">{body}</a>')
        else:
            cards += (f'<div class="mod-card off" style="{style}">'
                      f'{body}'
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


def page_atm():
    page_head("Module atmosphérique — ISA",
              "Propriétés de l'air en fonction de l'altitude et de l'écart "
              "de température ΔISA. La pression n'est pas affectée par "
              "ΔISA ; la masse volumique et la vitesse du son en découlent.",
              accent=ACCENTS["Atmosphère"][1])

    _init_slider("atm_h", 11700.0)
    _init_slider("atm_disa", 0.0)

    c_main, c_panel = ruban_saisie("atm_ruban")
    with c_main:
        with st.container(border=True,
                          height="stretch" if c_panel is not None else "content"):
            st.slider(r"Altitude $h$ [m]", 0.0, 20000.0, step=50.0,
                      key="atm_h_slider")
            h = float(st.session_state.atm_h_slider)
            st.caption(f"≈ {fr(h / FT)} ft")
            st.slider(r"Écart $\Delta_{ISA}$ [°C]", -30.0, 30.0, step=1.0,
                      key="atm_disa_slider")
            disa = float(st.session_state.atm_disa_slider)
            st.caption("plus froid que standard" if disa < 0 else
                       "plus chaud que standard" if disa > 0 else
                       "atmosphère standard")
    if c_panel is not None:
        with c_panel:
            carte_saisie([
                (r"Altitude $h$", 0.0, 20000.0,
                 {"m": (1.0, 100.0), "ft": (FT, 500.0)}, "atm_h"),
                (r"Écart $\Delta_{ISA}$ [°C]", -30.0, 30.0, 1.0, "atm_disa"),
            ])

    props = mod_atm.atmosphere(h, disa)

    with st.container(border=True):
        metrics_card("État de l'air au point courant", [
            metric(f"Température {sym('<i>T</i>')}", f"{props['T']:.2f}", "K",
                   f"{props['T'] - 273.15:+.1f} °C"),
            metric(f"Pression {sym('<i>P</i>')}", fr(props['P']), "Pa",
                   f"{props['P'] / 100:.1f} hPa"),
            metric(f"Masse volumique {sym('<i>ρ</i>')}",
                   f"{props['rho']:.4f}", "kg/m³",
                   f"{props['sigma'] * 100:.1f} % de ρ₀"),
            metric(f"Vitesse du son {sym('<i>a</i>')}",
                   f"{props['a']:.1f}", "m/s",
                   f"{props['a'] / KT:.0f} kt"),
        ], cols=2, accent=ACCENTS["Atmosphère"][0])

    ratios_strip([
        (sym('<i>θ</i> = <i>T</i>/<i>T</i><sub>0</sub>'),
         f"{props['theta']:.4f}"),
        (sym('<i>δ</i> = <i>P</i>/<i>P</i><sub>0</sub>'),
         f"{props['delta']:.4f}"),
        (sym('<i>σ</i> = <i>ρ</i>/<i>ρ</i><sub>0</sub>'),
         f"{props['sigma']:.4f}"),
    ], accent=ACCENTS["Atmosphère"][0])

    hs = np.linspace(0.0, 20000.0, 201)
    # (nom, unité, profil, valeur courante, texte de l'étiquette)
    profils = [
        ("Température T", "K", mod_atm.temperature(hs, disa), props['T'],
         f"{props['T']:.1f}"),
        ("Pression P", "Pa", mod_atm.pressure(hs, disa), props['P'],
         fr(props['P'])),
        ("Masse volumique ρ", "kg/m³", mod_atm.density(hs, disa), props['rho'],
         f"{props['rho']:.4f}"),
        ("Vitesse du son a", "m/s", mod_atm.speed_of_sound(hs, disa),
         props['a'], f"{props['a']:.1f}"),
    ]
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[f"{p[0]} [{p[1]}]" for p in profils],
                        horizontal_spacing=0.12, vertical_spacing=0.13)
    for i, (nom, unit, vals, cur, cur_txt) in enumerate(profils):
        row, col = i // 2 + 1, i % 2 + 1
        fig.add_trace(go.Scatter(x=vals, y=hs, mode="lines",
                                 line=dict(color=ACCENTS["Atmosphère"][0],
                                           width=2),
                                 showlegend=False,
                                 hovertemplate=f"{nom} : %{{x:.4g}} {unit}"
                                               f"<br>h : %{{y:,.0f}} m"
                                               f"<extra></extra>"),
                      row=row, col=col)
        fig.add_trace(go.Scatter(x=[cur], y=[h], mode="markers",
                                 marker=dict(color=RED, size=22, opacity=.15),
                                 showlegend=False, hoverinfo="skip"),
                      row=row, col=col)
        fig.add_trace(go.Scatter(x=[cur], y=[h], mode="markers+text",
                                 marker=dict(color=RED, size=9),
                                 text=[f" {cur_txt} {unit}"],
                                 textposition="middle right",
                                 textfont=dict(color=RED, size=12.5,
                                               family=FONT_MONO),
                                 cliponaxis=False,
                                 showlegend=False, hoverinfo="skip"),
                      row=row, col=col)
    fig.add_hline(y=mod_atm.H_TROPO, line_dash="dot", line_width=1.5,
                  line_color="#8B93A1", row="all", col="all")
    fig.add_annotation(text="Tropopause — 11 000 m", x=0.04,
                       xref="x domain", xanchor="left",
                       y=mod_atm.H_TROPO, yanchor="bottom", yshift=3,
                       showarrow=False,
                       font=dict(size=11, color="#8B93A1"),
                       row=1, col=1)
    fig.update_yaxes(title_text="h [m]", col=1)
    fig.update_layout(height=640, template="plotly_white",
                      font=dict(family=FONT_UI), margin=dict(t=40, b=40))
    with st.container(border=True):
        st.markdown('<div class="am-card-title">Profils atmosphériques '
                    '0 – 20 km</div>', unsafe_allow_html=True)
        st.caption(f"ΔISA = {disa:+.0f} °C · point courant à {fr(h)} m")
        st.plotly_chart(fig, config=PLOTLY_CONF)

    with st.expander("Altitude à partir de la pression"):
        p_query = st.number_input(r"Pression $P$ [Pa]", 5000.0, 110000.0,
                                  float(props['P']), 100.0)
        h_inv = mod_atm.altitude_from_pressure(p_query, disa)
        st.metric("Altitude h", f"{float(h_inv):,.1f} m")

    with st.expander("Formules du modèle ISA"):
        st.latex(r"T = T_0 + L\,h + \Delta T_{ISA}"
                 r"\qquad (h \le 11\,000\ \mathrm{m},\ L = -6.5\ "
                 r"\mathrm{K/km})")
        st.latex(r"P = P_0\left(\frac{T - \Delta T_{ISA}}{T_0}\right)"
                 r"^{-g/(R\,L)}\qquad\text{(pression indépendante de }"
                 r"\Delta T_{ISA})")
        st.latex(r"\rho = \frac{P}{R\,T}\qquad a = \sqrt{\gamma\,R\,T}")
        st.latex(r"\theta = \frac{T}{T_0}\qquad \delta = \frac{P}{P_0}"
                 r"\qquad \sigma = \frac{\rho}{\rho_0}")


def page_conv():
    page_head("Module de conversion — TAS / CAS / Mach",
              "Conversions isentropiques entre vitesse vraie, vitesse "
              "calibrée et nombre de Mach, basées sur l'atmosphère ISA.",
              accent=ACCENTS["Conversion"][1])

    convs = {
        "Mach → TAS": (mod_conv.mach_to_tas, "mach", "vitesse"),
        "TAS → Mach": (mod_conv.tas_to_mach, "vitesse", "mach"),
        "Mach → CAS": (mod_conv.mach_to_cas, "mach", "vitesse"),
        "CAS → Mach": (mod_conv.cas_to_mach, "vitesse", "mach"),
        "TAS → CAS": (mod_conv.tas_to_cas, "vitesse", "vitesse"),
        "CAS → TAS": (mod_conv.cas_to_tas, "vitesse", "vitesse"),
    }

    _init_slider("conv_h", 10000.0)
    _init_slider("conv_disa", 0.0)

    c_main, c_panel = ruban_saisie("conv_ruban")
    with c_main:
        with st.container(border=True,
                          height="stretch" if c_panel is not None else "content"):
            c1, c2 = st.columns([1, 2])
            choix = c1.radio("Conversion", list(convs))
            f, kind_in, kind_out = convs[choix]

            with c2:
                cc1, cc2 = st.columns(2)
                if kind_in == "mach":
                    val_in = cc1.number_input("Mach", 0.0, 1.0, 0.85, 0.01)
                    x = val_in
                    unite = cc2.radio("Unité de sortie", ["kt", "m/s"],
                                      horizontal=True)
                else:
                    unite = cc2.radio("Unité d'entrée/sortie", ["kt", "m/s"],
                                      horizontal=True)
                    val_in = cc1.number_input(f"Vitesse [{unite}]", 0.0,
                                              1200.0, 300.0, 1.0)
                    x = val_in * KT if unite == "kt" else val_in
                cc3, cc4 = st.columns(2)
                cc3.slider("Altitude h [m]", 0.0, 20000.0, step=50.0,
                           key="conv_h_slider")
                cc4.slider("ΔISA [°C]", -30.0, 30.0, step=1.0,
                           key="conv_disa_slider")
                h = float(st.session_state.conv_h_slider)
                disa = float(st.session_state.conv_disa_slider)
    if c_panel is not None:
        with c_panel:
            carte_saisie([
                ("Altitude h", 0.0, 20000.0,
                 {"m": (1.0, 100.0), "ft": (FT, 500.0)}, "conv_h"),
                ("ΔISA [°C]", -30.0, 30.0, 1.0, "conv_disa"),
            ])

    res = float(f(x, h, disa))
    with st.container(border=True):
        if kind_out == "mach":
            items = [metric(choix, f"{res:.4f}", "Mach",
                            f"{res * mod_atm.speed_of_sound(h, disa) / KT:.0f}"
                            f" kt TAS")]
        else:
            res_aff = res / KT if unite == "kt" else res
            autre = (f"{res:.2f} m/s" if unite == "kt"
                     else f"{res / KT:.2f} kt")
            items = [metric(choix, f"{res_aff:.2f}", unite, autre)]
        metrics_card("Résultat", items, cols=1,
                     accent=ACCENTS["Conversion"][0])

    hs = np.linspace(0.0, 20000.0, 201)
    ys = np.array([float(f(x, hh, disa)) for hh in hs])
    if kind_out == "vitesse" and unite == "kt":
        ys = ys / KT
    y_label = "Mach" if kind_out == "mach" else f"Vitesse [{unite}]"
    cur = res if kind_out == "mach" else (res / KT if unite == "kt" else res)
    cur_txt = f"{cur:.4f}" if kind_out == "mach" else f"{cur:.1f} {unite}"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ys, y=hs, mode="lines",
                             line=dict(color=ACCENTS["Conversion"][0],
                                       width=2),
                             showlegend=False,
                             hovertemplate=f"{y_label} : %{{x:.4g}}"
                                           f"<br>h : %{{y:,.0f}} m"
                                           f"<extra></extra>"))
    fig.add_trace(go.Scatter(x=[cur], y=[h], mode="markers",
                             marker=dict(color=RED, size=22, opacity=.15),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[cur], y=[h], mode="markers+text",
                             marker=dict(color=RED, size=9),
                             text=[f" {cur_txt}"], textposition="middle right",
                             textfont=dict(color=RED, size=12.5,
                                           family=FONT_MONO),
                             cliponaxis=False,
                             showlegend=False, hoverinfo="skip"))
    fig.update_layout(height=420, template="plotly_white", font=dict(family=FONT_UI),
                      xaxis_title=y_label, yaxis_title="h [m]",
                      margin=dict(t=40, b=40))
    with st.container(border=True):
        st.markdown(f'<div class="am-card-title">{choix} en fonction de '
                    'l\'altitude</div>', unsafe_allow_html=True)
        st.caption(f"Entrée constante : {val_in:g} "
                   f"{'Mach' if kind_in == 'mach' else unite} · "
                   f"ΔISA = {disa:+.0f} °C")
        st.plotly_chart(fig, config=PLOTLY_CONF)

    with st.expander("Formules de conversion"):
        st.latex(r"V_{TAS} = M\,a_0\sqrt{\theta}")
        st.latex(r"V_{CAS} = a_0\sqrt{5\left\{\left[\delta\left("
                 r"(1+0.2M^2)^{3.5}-1\right)+1\right]^{1/3.5}-1\right\}}")
        st.latex(r"M = \sqrt{5\left\{\left[\frac{1}{\delta}\left(\left["
                 r"1+0.2\left(\frac{V_{CAS}}{a_0}\right)^{2}\right]^{3.5}"
                 r"-1\right)+1\right]^{1/3.5}-1\right\}}")
        st.caption("TAS ↔ CAS : passage obligatoire par le nombre de Mach.")


def page_aero():
    page_head("Module aérodynamique — A380",
              "Coefficients de portance, de traînée et de moment construits "
              "sur les données OpenVSP / VSPAERO — aile-fuselage, empennage "
              "horizontal et avion complet.",
              accent=ACCENTS["Aérodynamique"][1])

    model = load_aero_model()
    grid = model['f_clwb']
    a_min, a_max = float(grid['x_alpha'][0]), float(grid['x_alpha'][-1])
    m_min, m_max = float(grid['y_mach'][0]), float(grid['y_mach'][-1])

    _init_slider("aero_alpha", 2.0)
    _init_slider("aero_mach", min(0.85, m_max))
    _init_slider("aero_dit", 0.0)

    c_main, c_panel = ruban_saisie("aero_ruban")
    with c_main:
        with st.container(border=True,
                          height="stretch" if c_panel is not None else "content"):
            c1, c2, c3 = st.columns(3)
            c1.slider(r"Angle d'attaque $\alpha$ [deg]", a_min, a_max,
                      step=0.1, key="aero_alpha_slider")
            c2.slider(r"Mach $M$", m_min, m_max, step=0.01,
                      key="aero_mach_slider")
            c3.slider(r"Calage empennage $\delta_{it}$ [deg]", -10.0, 10.0,
                      step=0.5, key="aero_dit_slider")
            alpha = float(st.session_state.aero_alpha_slider)
            mach = float(st.session_state.aero_mach_slider)
            dit = float(st.session_state.aero_dit_slider)
    if c_panel is not None:
        with c_panel:
            carte_saisie([
                (r"$\alpha$ [deg]", a_min, a_max, 0.1, "aero_alpha"),
                (r"Mach $M$", m_min, m_max, 0.01, "aero_mach"),
                (r"$\delta_{it}$ [deg]", -10.0, 10.0, 0.5, "aero_dit"),
            ])

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
    cl_t, cd_t = rows[2][1], rows[2][2]

    with st.container(border=True):
        for nom, cl, cd, cm in rows:
            metrics_card(nom, [
                metric(sym('<i>C</i><sub>L</sub>'), f"{cl:.4f}"),
                metric(sym('<i>C</i><sub>D</sub>'), f"{cd:.5f}"),
                metric(sym('<i>C</i><sub>M</sub>'), f"{cm:.4f}"),
            ], cols=3, small=True,
                         accent=ACCENTS["Aérodynamique"][0])

    ratios_strip([
        (f"Downwash {sym('<i>ε</i>')}", f"{eps:.3f}°"),
        (sym('<i>α</i><sub>ht</sub> = <i>α</i> − <i>ε</i> + '
             '<i>δ</i><sub>it</sub>'),
         f"{alpha - eps + dit:.3f}°"),
        (f"Finesse {sym('<i>C</i><sub>L,t</sub>/<i>C</i><sub>D,t</sub>')}",
         f"{cl_t / cd_t:.2f}" if cd_t else "—"),
    ], accent=ACCENTS["Aérodynamique"][0])

    tab1, tab2, tab3 = st.tabs(["Coefficients vs α", "Polaire", "Surface 3D"])

    curves = aero_curves(dit, mach)
    alphas = curves['alpha']

    with tab1:
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=("CL", "CD", "CM"))
        for i, key in enumerate(("cl", "cd", "cm"), start=1):
            cname = key.upper()
            fig.add_trace(go.Scatter(x=alphas, y=curves[f'{key}_t'],
                                     name="total", legendgroup="t",
                                     showlegend=(i == 1),
                                     line=dict(
                                         color=ACCENTS["Aérodynamique"][0],
                                         width=2),
                                     hovertemplate=f"α : %{{x:.2f}}°<br>"
                                                   f"{cname} total : %{{y:.4f}}"
                                                   f"<extra></extra>"),
                          row=1, col=i)
            fig.add_trace(go.Scatter(x=alphas, y=curves[f'{key}_wb'],
                                     name="WB seul", legendgroup="wb",
                                     showlegend=(i == 1),
                                     line=dict(color="#8B93A1", dash="dash"),
                                     hovertemplate=f"α : %{{x:.2f}}°<br>"
                                                   f"{cname} WB : %{{y:.4f}}"
                                                   f"<extra></extra>"),
                          row=1, col=i)
            cur = {"cl": rows[2][1], "cd": rows[2][2], "cm": rows[2][3]}[key]
            fmt = ".5f" if key == "cd" else ".4f"
            fig.add_trace(go.Scatter(x=[alpha], y=[cur], mode="markers+text",
                                     marker=dict(color=RED, size=9),
                                     text=[f" {cur:{fmt}}"],
                                     textposition="middle right",
                                     textfont=dict(color=RED, size=11.5,
                                                   family=FONT_MONO),
                                     cliponaxis=False,
                                     showlegend=False, hoverinfo="skip"),
                          row=1, col=i)
        fig.update_xaxes(title_text="α [deg]")
        fig.update_layout(height=420, template="plotly_white", font=dict(family=FONT_UI),
                          title=f"Coefficients totaux — M = {mach:.2f}, "
                                f"δ_it = {dit:+.1f}°")
        st.plotly_chart(fig, config=PLOTLY_CONF)

    with tab2:
        cl_arr = np.array(curves['cl_t'])
        cd_arr = np.array(curves['cd_t'])
        with np.errstate(divide="ignore", invalid="ignore"):
            fin = np.where(cd_arr > 0, cl_arr / cd_arr, -np.inf)
        idx = int(np.argmax(fin))
        cl_opt, cd_opt, fin_opt = cl_arr[idx], cd_arr[idx], fin[idx]
        acc_vif = ACCENTS["Aérodynamique"][1]
        fig = go.Figure()
        # tangente depuis l'origine : sa pente est la finesse maximale
        fig.add_trace(go.Scatter(x=[0, cd_opt * 1.12], y=[0, cl_opt * 1.12],
                                 mode="lines",
                                 line=dict(color="#8B93A1", dash="dot",
                                           width=1),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=curves['cd_t'], y=curves['cl_t'],
                                 mode="lines",
                                 line=dict(
                                     color=ACCENTS["Aérodynamique"][0],
                                     width=2),
                                 showlegend=False,
                                 hovertemplate="CD : %{x:.5f}<br>"
                                               "CL : %{y:.4f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[cd_opt], y=[cl_opt], mode="markers+text",
                                 marker=dict(color=acc_vif, size=13,
                                             symbol="star"),
                                 text=[f"  finesse max {fin_opt:.1f}"],
                                 textposition="middle right",
                                 textfont=dict(color=acc_vif, size=12.5,
                                               family=FONT_MONO),
                                 cliponaxis=False, showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[rows[2][2]], y=[rows[2][1]],
                                 mode="markers",
                                 marker=dict(color=RED, size=9),
                                 name="point courant", showlegend=False,
                                 hovertemplate="CD : %{x:.5f}<br>"
                                               "CL : %{y:.4f}<extra>"
                                               "point courant</extra>"))
        fig.update_layout(height=420, template="plotly_white", font=dict(family=FONT_UI),
                          xaxis_title="CD_t", yaxis_title="CL_t",
                          title=f"Polaire avion complet — M = {mach:.2f}, "
                                f"δ_it = {dit:+.1f}°")
        st.plotly_chart(fig, config=PLOTLY_CONF)

    with tab3:
        coef = st.selectbox("Coefficient",
                            ["CL_t", "CL_wb", "CL_ht",
                             "CD_t", "CD_wb", "CD_ht",
                             "CM_t", "CM_wb", "CM_ht"])
        a_s, m_s, z = aero_surface(coef, dit)
        # rows : [0]=WB, [1]=HT, [2]=total ; chaque ligne (nom, CL, CD, CM)
        z_cur = {"CL_t": rows[2][1], "CD_t": rows[2][2], "CM_t": rows[2][3],
                 "CL_wb": rows[0][1], "CD_wb": rows[0][2], "CM_wb": rows[0][3],
                 "CL_ht": rows[1][1], "CD_ht": rows[1][2],
                 "CM_ht": rows[1][3]}[coef]
        fig = go.Figure(go.Surface(x=m_s, y=a_s, z=z, colorscale=SCALE_AERO,
                                   hovertemplate="Mach : %{x:.2f}<br>"
                                                 "α : %{y:.2f}°<br>"
                                                 f"{coef} : %{{z:.4f}}"
                                                 "<extra></extra>"))
        fig.add_trace(go.Scatter3d(x=[mach], y=[alpha], z=[z_cur],
                                   mode="markers+text",
                                   marker=dict(color=RED, size=5),
                                   text=[f"  {z_cur:.4f}"],
                                   textfont=dict(color=RED, size=12),
                                   name="point courant", showlegend=False))
        fig.update_layout(height=560, template="plotly_white", font=dict(family=FONT_UI),
                          scene=dict(xaxis_title="Mach",
                                     yaxis_title="α [deg]",
                                     zaxis_title=coef),
                          title=f"{coef}(α, M) — δ_it = {dit:+.1f}°")
        st.plotly_chart(fig, config=PLOTLY_CONF)

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


def page_prop():
    page_head("Module de propulsion & émissions — Trent 970",
              "Poussée nette et débit carburant par polynômes de surface "
              "calibrés ; indices d'émission OACI généralisés aux conditions "
              "de vol par la méthode Boeing Fuel Flow.",
              accent=ACCENTS["Propulsion & Émissions"][1])

    _init_slider("prop_n1", 90.0)
    _init_slider("prop_mach", 0.85)
    _init_slider("prop_h", 10668.0)
    _init_slider("prop_disa", 0.0)

    c_main, c_panel = ruban_saisie("prop_ruban")
    with c_main:
        with st.container(border=True,
                          height="stretch" if c_panel is not None else "content"):
            c1, c2 = st.columns(2)
            c1.slider(r"Fan speed $N1$ [%]", 60.0, 100.0, step=0.5,
                      key="prop_n1_slider")
            c2.slider(r"Mach $M$", 0.0, 0.85, step=0.01,
                      key="prop_mach_slider")
            c1.slider(r"Altitude $h$ [m]", 0.0, 15000.0, step=50.0,
                      key="prop_h_slider")
            c2.slider(r"$\Delta_{ISA}$ [°C]", -30.0, 30.0, step=1.0,
                      key="prop_disa_slider")
            n1 = float(st.session_state.prop_n1_slider)
            mach = float(st.session_state.prop_mach_slider)
            h = float(st.session_state.prop_h_slider)
            disa = float(st.session_state.prop_disa_slider)
    if c_panel is not None:
        with c_panel:
            carte_saisie([
                (r"$N1$ [%]", 60.0, 100.0, 0.5, "prop_n1"),
                (r"Mach $M$", 0.0, 0.85, 0.01, "prop_mach"),
                (r"Altitude $h$", 0.0, 15000.0,
                 {"m": (1.0, 100.0), "ft": (FT, 500.0)}, "prop_h"),
                (r"$\Delta_{ISA}$ [°C]", -30.0, 30.0, 1.0, "prop_disa"),
            ])

    fn = float(mod_prop.get_thrust(n1, mach, h, disa))
    ei = mod_prop.get_emission_indices(n1, mach, h, disa)
    wf = float(ei['WF'])
    theta = float(mod_atm.theta(h, disa))
    delta = float(mod_atm.delta(h, disa))

    with st.container(border=True):
        metrics_card("Performances moteur", [
            metric(f"Poussée nette {sym('<i>F</i><sub>N</sub>')}",
                   f"{fn:.4f}", "",
                   f"× 4 moteurs = {4 * fn:.4f}"),
            metric(f"Débit carburant {sym('<i>W</i><sub>F</sub>')}",
                   f"{wf:.4f}", "kg/s",
                   f"{fr(wf * 3600)} kg/h"),
            metric(sym('<i>W</i><sub>F,C</sub><sup>REF</sup>') + " (BFF)",
                   f"{ei['WF_C_REF']:.4f}", "kg/s",
                   "table OACI : 0.261 – 2.738"),
        ], cols=3, small=True,
            accent=ACCENTS["Propulsion & Émissions"][0])

    with st.container(border=True):
        metrics_card("Indices d'émission généralisés (méthode Boeing "
                     "Fuel Flow)", [
            metric(sym('EI<sub>NOx</sub>'), f"{ei['EI_NOx']:.3f}", "g/kg",
                   f"{ei['EI_NOx'] * wf:.3f} g/s"),
            metric(sym('EI<sub>UHC</sub>'), f"{ei['EI_UHC']:.4f}", "g/kg",
                   f"{ei['EI_UHC'] * wf:.4f} g/s"),
            metric(sym('EI<sub>CO</sub>'), f"{ei['EI_CO']:.3f}", "g/kg",
                   f"{ei['EI_CO'] * wf:.3f} g/s"),
            metric(sym('EI<sub>CO₂</sub>'), f"{ei['EI_CO2']:.2f}", "kg/kg",
                   f"{ei['EI_CO2'] * wf:.3f} kg/s"),
        ], cols=4, small=True,
            accent=ACCENTS["Propulsion & Émissions"][0])

    ratios_strip([
        (sym('<i>N</i>1<sub>cor</sub> = <i>N</i>1/√<i>θ</i>'),
         f"{n1 / np.sqrt(theta):.2f} %"),
        (sym('<i>W</i><sub>F,C</sub> = <i>W</i><sub>F</sub>/'
             '(<i>δ</i>√<i>θ</i>)'),
         f"{wf / (delta * np.sqrt(theta)):.4f} kg/s"),
        (f"Humidité spécifique {sym('<i>ω</i>')}", f"{ei['omega']:.6f}"),
    ], accent=ACCENTS["Propulsion & Émissions"][0])

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
        ], cols=5, small=True,
            accent=ACCENTS["Propulsion & Émissions"][0])

    with st.expander("Formules du modèle (méthode Boeing Fuel Flow)"):
        st.latex(r"F_N = \bar f_{th}\!\left(\tfrac{N1}{\sqrt{\theta}},\,M"
                 r"\right)\delta\qquad "
                 r"W_F = \bar f_{wf}\!\left(\tfrac{N1}{\sqrt{\theta}},\,M"
                 r"\right)\delta\sqrt{\theta}")
        st.latex(r"W_{F,C}^{REF} = \frac{(1+0.2M^2)\,\theta^{\,3.8+y}}"
                 r"{\delta^{\,1-x}}\,W_{F,C}\qquad x=1,\ y=0.5"
                 r"\ \text{(Ghazi et Botez)}")
        st.latex(r"EI_{UHC} = EI_{UHC,C}^{REF}\,\theta^{3.3}/\delta^{1.02}"
                 r"\qquad EI_{CO} = EI_{CO,C}^{REF}\,\theta^{3.3}/"
                 r"\delta^{1.02}")
        st.latex(r"EI_{NO_x} = EI_{NO_x,C}^{REF}\sqrt{\delta^{1.02}/"
                 r"\theta^{3.3}}\;e^{-19\,(\omega - 0.00634)}")
        st.latex(r"\omega = \frac{0.62197058\;RH\;P_{SAT}}"
                 r"{0.1P - RH\;P_{SAT}}\qquad "
                 r"P_{SAT} = 6.107\times 10^{\frac{7.5\,(T-273.15)}"
                 r"{T-35.85}}\qquad RH = 0.80")
        st.latex(r"\Delta m_i = EI_i\,\Delta F_B\qquad "
                 r"EI_{CO_2} = 3.16\ \mathrm{kg/kg}")

    tab1, tab2, tab3 = st.tabs(["FN / WF vs N1", "Surfaces 3D", "Émissions vs N1"])

    n1s = np.linspace(60.0, 100.0, 81)
    machs = np.linspace(0.0, 0.85, 6)

    with tab1:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Poussée FN", "Débit WF [kg/s]"))
        for M in machs:
            fig.add_trace(go.Scatter(x=n1s,
                                     y=mod_prop.get_thrust(n1s, M, h, disa),
                                     name=f"M={M:.2f}", legendgroup=f"{M:.2f}",
                                     showlegend=True,
                                     hovertemplate="N1 : %{x:.1f} %<br>"
                                                   "FN : %{y:.3f}"
                                                   "<extra>M=" + f"{M:.2f}"
                                                   "</extra>"), row=1, col=1)
            fig.add_trace(go.Scatter(x=n1s,
                                     y=mod_prop.get_fuel_flow(n1s, M, h, disa),
                                     name=f"M={M:.2f}", legendgroup=f"{M:.2f}",
                                     showlegend=False,
                                     hovertemplate="N1 : %{x:.1f} %<br>"
                                                   "WF : %{y:.4f} kg/s"
                                                   "<extra>M=" + f"{M:.2f}"
                                                   "</extra>"), row=1, col=2)
        fig.add_trace(go.Scatter(x=[n1], y=[fn], mode="markers+text",
                                 marker=dict(color=RED, size=9),
                                 text=[f"  {fn:.3f}"], textposition="top center",
                                 textfont=dict(color=RED, size=12,
                                               family=FONT_MONO),
                                 cliponaxis=False, showlegend=False,
                                 hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[n1], y=[wf], mode="markers+text",
                                 marker=dict(color=RED, size=9),
                                 text=[f"  {wf:.3f}"], textposition="top center",
                                 textfont=dict(color=RED, size=12,
                                               family=FONT_MONO),
                                 cliponaxis=False, showlegend=False,
                                 hoverinfo="skip"), row=1, col=2)
        fig.update_xaxes(title_text="N1 [%]")
        fig.update_layout(height=440, template="plotly_white", font=dict(family=FONT_UI),
                          colorway=APPLE_SEQ,
                          title=f"h = {h:.0f} m  |  ΔISA = {disa:+.0f} °C")
        st.plotly_chart(fig, config=PLOTLY_CONF)

    with tab2:
        machs_3d = np.linspace(0.0, 0.85, 40)
        N1g, Mg = np.meshgrid(n1s, machs_3d)
        fn_surf = mod_prop.get_thrust(N1g, Mg, h, disa)
        wf_surf = mod_prop.get_fuel_flow(N1g, Mg, h, disa)
        cc1, cc2 = st.columns(2)
        for col, (surf, nom, z_cur) in zip(
                (cc1, cc2),
                ((fn_surf, "FN", fn), (wf_surf, "WF [kg/s]", wf))):
            fig = go.Figure(go.Surface(x=N1g, y=Mg, z=surf,
                                       colorscale=SCALE_PROP,
                                       hovertemplate="N1 : %{x:.1f} %<br>"
                                                     "Mach : %{y:.2f}<br>"
                                                     f"{nom} : %{{z:.4f}}"
                                                     "<extra></extra>"))
            fig.add_trace(go.Scatter3d(x=[n1], y=[mach], z=[z_cur],
                                       mode="markers+text",
                                       marker=dict(color=RED, size=5),
                                       text=[f"  {z_cur:.3f}"],
                                       textfont=dict(color=RED, size=12),
                                       name="point courant", showlegend=False))
            fig.update_layout(height=480, template="plotly_white", font=dict(family=FONT_UI),
                              scene=dict(xaxis_title="N1 [%]",
                                         yaxis_title="Mach",
                                         zaxis_title=nom),
                              title=nom)
            col.plotly_chart(fig, config=PLOTLY_CONF)

    with tab3:
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=("EI NOx [g/kg]", "EI CO [g/kg]",
                                            "EI UHC [g/kg]"))
        for M in machs:
            e = mod_prop.get_emission_indices(n1s, M, h, disa)
            for i, key in enumerate(("EI_NOx", "EI_CO", "EI_UHC"), start=1):
                fig.add_trace(go.Scatter(x=n1s, y=e[key],
                                         name=f"M={M:.2f}",
                                         legendgroup=f"{M:.2f}",
                                         showlegend=(i == 1),
                                         hovertemplate="N1 : %{x:.1f} %<br>"
                                                       + key + " : %{y:.3f}"
                                                       " g/kg<extra>M="
                                                       + f"{M:.2f}</extra>"),
                              row=1, col=i)
        for i, key in enumerate(("EI_NOx", "EI_CO", "EI_UHC"), start=1):
            fig.add_trace(go.Scatter(x=[n1], y=[ei[key]], mode="markers+text",
                                     marker=dict(color=RED, size=9),
                                     text=[f"  {ei[key]:.2f}"],
                                     textposition="top center",
                                     textfont=dict(color=RED, size=11.5,
                                                   family=FONT_MONO),
                                     cliponaxis=False, showlegend=False,
                                     hoverinfo="skip"), row=1, col=i)
        fig.update_xaxes(title_text="N1 [%]")
        fig.update_layout(height=440, template="plotly_white", font=dict(family=FONT_UI),
                          colorway=APPLE_SEQ,
                          title=f"Indices d'émission — h = {h:.0f} m  |  "
                                f"ΔISA = {disa:+.0f} °C")
        st.plotly_chart(fig, config=PLOTLY_CONF)


def page_trim():
    page_head("Module d'équilibrage (Trim)",
              accent=ACCENTS["Équilibrage (Trim)"][1])
    st.info("Module en développement — à venir : équilibrage longitudinal "
            "de l'avion à partir des modules aérodynamique et propulsion.")


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
}

@st.cache_data
def _img_b64(path, mtime):
    """Image encodée en base64 — mtime invalide le cache si le fichier change."""
    return base64.b64encode(Path(path).read_bytes()).decode()


def apply_page_background():
    """Image cockpit en fond de toutes les pages, sous un voile clair.

    Les cartes passent en blanc translucide + flou (verre dépoli) pour
    rester lisibles. CSS limité à des propriétés de fond — dégradation
    sans risque si la structure DOM change.
    """
    if not BG_IMG.exists():
        return
    b64 = _img_b64(str(BG_IMG), BG_IMG.stat().st_mtime)
    # image bien présente à l'accueil, simple filigrane dans les modules
    voile = .60 if st.session_state.get("nav", "Accueil") == "Accueil" else .90
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(rgba(247, 249, 252, {voile}),
                                    rgba(247, 249, 252, {voile})),
                    url("data:image/jpeg;base64,{b64}") center / cover
                    no-repeat fixed;
    }}
    [data-testid="stHeader"] {{ background: transparent; }}
    /* cartes en verre dépoli pour la lisibilité sur l'image */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(255, 255, 255, .95);
        -webkit-backdrop-filter: blur(8px);
        backdrop-filter: blur(8px);
    }}
    </style>
    """, unsafe_allow_html=True)


def apply_sidebar_background():
    """Image de flux d'air en fond de sidebar, texte passé en blanc.

    CSS volontairement confiné à [data-testid="stSidebar"] : on ne touche
    ni aux polices ni à la visibilité d'éléments (cf. mésaventure des
    icônes Material avec le CSS global).
    """
    if not SIDEBAR_IMG.exists():
        return
    b64 = _img_b64(str(SIDEBAR_IMG), SIDEBAR_IMG.stat().st_mtime)
    # pastille de l'item actif : dégradé aux couleurs du module sélectionné
    fonce, vif = ACCENTS.get(st.session_state.get("nav", "Accueil"),
                             (NAVY, "#3E6B99"))
    st.markdown(f"""
    <style>
    [data-testid="stSidebar"] {{
        background: linear-gradient(rgba(9, 21, 38, .68),
                                    rgba(9, 21, 38, .68)),
                    url("data:image/jpeg;base64,{b64}") center / cover
                    no-repeat;
    }}
    [data-testid="stSidebarContent"] {{ background: transparent !important; }}
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {{ color: #FFFFFF !important; }}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
        color: rgba(255, 255, 255, .78) !important;
    }}
    /* bloc marque + label MODULE (couleurs pour fond sombre) */
    .sb-title {{ color: #FFFFFF; }}
    .sb-sub {{ color: rgba(255, 255, 255, .75); }}
    .sb-label {{ color: rgba(255, 255, 255, .55); }}
    /* items de nav : pastille translucide sur l'item actif + survol */
    [data-testid="stSidebar"] label[data-baseweb="radio"] {{
        width: 100%; padding: 8px 12px; margin: 1px 0; border-radius: 8px;
        transition: background .12s;
    }}
    [data-testid="stSidebar"] label[data-baseweb="radio"]:hover {{
        background: rgba(255, 255, 255, .08);
    }}
    [data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {{
        background: linear-gradient(135deg, {fonce}, {vif});
    }}
    [data-testid="stSidebar"] [role="radiogroup"] p {{
        font-size: 14.5px; font-weight: 600;
    }}
    [data-testid="stSidebar"] label[data-baseweb="radio"]:not(:has(input:checked)) p {{
        color: rgba(255, 255, 255, .72) !important;
    }}
    /* pied de sidebar : pousser le dernier élément (caption) tout en bas */
    [data-testid="stSidebarContent"] {{ display: flex; flex-direction: column; }}
    [data-testid="stSidebarUserContent"] {{
        flex: 1 1 auto; display: flex; flex-direction: column;
    }}
    [data-testid="stSidebarUserContent"] > div {{ flex: 1; }}
    [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] {{
        height: 100%;
    }}
    [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"]
    > div:last-child {{
        margin-top: auto; padding-top: 14px;
        border-top: 1px solid rgba(255, 255, 255, .18);
    }}
    </style>
    """, unsafe_allow_html=True)


# Navigation par les cartes de l'accueil : ?page=<module> dans l'URL
if "page" in st.query_params:
    _cible = st.query_params["page"]
    if _cible in PAGES:
        st.session_state.nav = _cible
    st.query_params.clear()

apply_page_background()
apply_sidebar_background()
st.sidebar.markdown("""
<style>
.sb-brand { display: flex; align-items: center; gap: 12px; margin: 4px 0 18px; }
.sb-mark { width: 44px; height: 44px; border-radius: 11px; background: #1B3A5C;
           border: 1px solid rgba(255, 255, 255, .25); color: #fff;
           display: grid; place-items: center; font-weight: 700;
           font-size: 14px; letter-spacing: .02em; }
.sb-title { font-size: 16px; font-weight: 700; letter-spacing: -.01em; }
.sb-sub { font-size: 12.5px; margin-top: 1px; }
.sb-label { font-size: 11px; font-weight: 700; text-transform: uppercase;
            letter-spacing: .09em; margin: 0 0 4px 2px; }
</style>
<div class="sb-brand">
  <div class="sb-mark">380</div>
  <div>
    <div class="sb-title">A380 — MGA803</div>
    <div class="sb-sub">Performances avion</div>
  </div>
</div>
<div class="sb-label">Module</div>
""", unsafe_allow_html=True)
choix_page = st.sidebar.radio("Module", list(PAGES), key="nav",
                              label_visibility="collapsed")
st.sidebar.caption("Analyse et optimisation des performances des avions — "
                   "ÉTS, É2026")

PAGES[choix_page]()
