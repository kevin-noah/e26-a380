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
.am-value { font-family: "Source Code Pro", ui-monospace, SFMono-Regular,
            monospace; font-variant-numeric: tabular-nums; font-size: 42px;
            font-weight: 600; color: #1A2230; letter-spacing: -.01em;
            line-height: 1.15; margin-top: 2px; display: flex;
            align-items: baseline; gap: 8px; }
.am-grid.sm .am-value { font-size: 26px; }
.am-unit { font-size: .45em; font-weight: 600; color: #8B93A1; }
.am-pill { display: inline-block; margin-top: 8px;
           font-family: "Source Code Pro", ui-monospace, monospace;
           font-size: 13px; font-weight: 600; color: #5B6573;
           background: #F4F6F9; border: 1px solid #E7EBF0;
           border-radius: 99px; padding: 2px 12px; }
.am-ratios { display: grid; background: #F7F9FC; border: 1px solid #E4E8EE;
             border-radius: 10px; padding: 12px 24px; }
.am-ratio { display: flex; align-items: baseline; gap: 10px; }
.am-ratio + .am-ratio { border-left: 1px solid #E4E8EE; padding-left: 24px; }
.am-rlabel { font-size: 13.5px; color: #5B6573; }
.am-rvalue { font-family: "Source Code Pro", ui-monospace, monospace;
             font-size: 19px; font-weight: 600; color: #1A2230; }
</style>
""", unsafe_allow_html=True)


def fr(x, dec=0):
    """Format numérique : espace pour les milliers (style français)."""
    return f"{x:,.{dec}f}".replace(",", " ")


def page_head(titre, lede=""):
    lede_html = f'<p class="am-lede">{lede}</p>' if lede else ""
    st.markdown(f'<h1 class="am-h1">{titre}</h1>{lede_html}',
                unsafe_allow_html=True)


def metric(label, value, unit="", pill=None):
    u = f'<span class="am-unit">{unit}</span>' if unit else ""
    p = f'<div><span class="am-pill">{pill}</span></div>' if pill else ""
    return (f'<div class="am-metric"><div class="am-mlabel">{label}</div>'
            f'<div class="am-value">{value}{u}</div>{p}</div>')


def metrics_card(title, items, cols=2, small=False):
    """Grille de métriques façon maquette — à utiliser dans un container."""
    cls = f"am-grid cols-{cols}" + (" sm" if small else "")
    st.markdown(f'<div class="am-card-title">{title}</div>'
                f'<div class="{cls}">{"".join(items)}</div>',
                unsafe_allow_html=True)


def ratios_strip(pairs):
    """Bandeau gris de grandeurs intermédiaires — pairs = [(label, val)]."""
    inner = "".join(f'<div class="am-ratio"><span class="am-rlabel">{l}</span>'
                    f'<span class="am-rvalue">{v}</span></div>'
                    for l, v in pairs)
    st.markdown(f'<div class="am-ratios" style="grid-template-columns:'
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
    f = {'CL_t': mod_aero.get_cl_total,
         'CD_t': mod_aero.get_cd_total,
         'CM_t': mod_aero.get_cm_total}[coef]
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
body {{ margin: 0; font-family: "Source Sans Pro", -apple-system,
        BlinkMacSystemFont, sans-serif; }}
.hero {{ position: relative; height: 340px; border-radius: 12px;
         overflow: hidden; }}
.hero video {{ position: absolute; inset: 0; width: 100%; height: 100%;
               object-fit: cover; }}
.hero .hero-shade {{ position: absolute; inset: 0; background:
  linear-gradient(180deg, rgba(8,16,28,.10) 0%, rgba(8,16,28,.60) 100%); }}
.hero .hero-text {{ position: absolute; left: 36px; bottom: 26px; color: #fff; }}
.hero .hero-text h1 {{ font-size: 46px; font-weight: 700; margin: 0;
                       letter-spacing: -.02em; line-height: 1.1; }}
.hero .hero-text p {{ margin: 6px 0 0; font-size: 16px; opacity: .88; }}
</style>
<div class="hero">
  <video autoplay loop muted playsinline
         src="data:video/mp4;base64,{b64}"></video>
  <div class="hero-shade"></div>
  <div class="hero-text">
    <h1>Airbus A380</h1>
    <p>MGA803 — Analyse et optimisation des performances des avions · ÉTS, É2026</p>
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


# CSS scopé aux classes .mod-grid / .mod-card uniquement (cartes accueil)
_CARDS_CSS = """
<style>
.mod-grid { display: grid; grid-template-columns: repeat(2, 1fr);
            gap: 18px; margin-top: 10px; }
.mod-card { display: block; position: relative; background: #fff;
            border: 1px solid #E4E8EE; border-radius: 12px;
            padding: 22px 26px; text-decoration: none !important;
            transition: border-color .15s, box-shadow .15s, transform .15s; }
a.mod-card:hover { border-color: #1B3A5C; transform: translateY(-2px);
                   box-shadow: 0 6px 18px rgba(16, 24, 40, .08); }
.mod-card .mod-head { display: flex; justify-content: space-between;
                      align-items: center; }
.mod-card .mod-num { font-family: ui-monospace, SFMono-Regular, monospace;
                     font-size: 13px; font-weight: 600; color: #8B93A1; }
.mod-card .mod-arrow { color: #5B6573; font-size: 19px; line-height: 1; }
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
        if dispo:
            cards += (f'<a class="mod-card" target="_self" '
                      f'href="?page={quote(nom)}">{body}</a>')
        else:
            cards += (f'<div class="mod-card off">{body}'
                      f'<span class="mod-pill">À VENIR</span></div>')
    st.markdown(_CARDS_CSS + f'<div class="mod-grid">{cards}</div>',
                unsafe_allow_html=True)


def _h_from_slider():
    st.session_state.atm_h_input = st.session_state.atm_h_slider


def _h_from_input():
    st.session_state.atm_h_slider = st.session_state.atm_h_input


def page_atm():
    page_head("Module atmosphérique — ISA",
              "Propriétés de l'air en fonction de l'altitude et de l'écart "
              "de température ΔISA. La pression n'est pas affectée par "
              "ΔISA ; la masse volumique et la vitesse du son en découlent.")

    st.session_state.setdefault("atm_h_slider", 11700.0)
    st.session_state.setdefault("atm_h_input", 11700.0)

    with st.container(border=True):
        c1, c2 = st.columns([2.8, 1], vertical_alignment="bottom")
        c1.slider("Altitude h [m]", 0.0, 20000.0, step=50.0,
                  key="atm_h_slider", on_change=_h_from_slider)
        c2.number_input("Saisie directe [m]", 0.0, 20000.0, step=100.0,
                        key="atm_h_input", on_change=_h_from_input)
        h = float(st.session_state.atm_h_slider)
        c1.caption(f"≈ {fr(h / FT)} ft")
        disa = st.slider("Écart ΔISA [°C]", -30.0, 30.0, 0.0, 1.0)
        st.caption("plus froid que standard" if disa < 0 else
                   "plus chaud que standard" if disa > 0 else
                   "atmosphère standard")

    props = mod_atm.atmosphere(h, disa)

    with st.container(border=True):
        metrics_card("État de l'air au point courant", [
            metric("Température T", f"{props['T']:.2f}", "K",
                   f"{props['T'] - 273.15:+.1f} °C"),
            metric("Pression P", fr(props['P']), "Pa",
                   f"{props['P'] / 100:.1f} hPa"),
            metric("Masse volumique ρ", f"{props['rho']:.4f}", "kg/m³",
                   f"{props['sigma'] * 100:.1f} % de ρ₀"),
            metric("Vitesse du son a", f"{props['a']:.1f}", "m/s",
                   f"{props['a'] / KT:.0f} kt"),
        ], cols=2)

    ratios_strip([("θ = T/T₀", f"{props['theta']:.4f}"),
                  ("δ = P/P₀", f"{props['delta']:.4f}"),
                  ("σ = ρ/ρ₀", f"{props['sigma']:.4f}")])

    hs = np.linspace(0.0, 20000.0, 201)
    profils = [
        ("T [K]", mod_atm.temperature(hs, disa), props['T']),
        ("P [Pa]", mod_atm.pressure(hs, disa), props['P']),
        ("ρ [kg/m³]", mod_atm.density(hs, disa), props['rho']),
        ("a [m/s]", mod_atm.speed_of_sound(hs, disa), props['a']),
    ]
    fig = make_subplots(rows=1, cols=4, shared_yaxes=True,
                        subplot_titles=[p[0] for p in profils])
    for i, (_, vals, cur) in enumerate(profils, start=1):
        fig.add_trace(go.Scatter(x=vals, y=hs, mode="lines",
                                 line=dict(color=NAVY, width=2),
                                 showlegend=False), row=1, col=i)
        fig.add_trace(go.Scatter(x=[cur], y=[h], mode="markers",
                                 marker=dict(color=RED, size=22, opacity=.15),
                                 showlegend=False, hoverinfo="skip"),
                      row=1, col=i)
        fig.add_trace(go.Scatter(x=[cur], y=[h], mode="markers",
                                 marker=dict(color=RED, size=9),
                                 showlegend=False), row=1, col=i)
    fig.update_yaxes(title_text="h [m]", row=1, col=1)
    fig.update_layout(height=420, template="plotly_white",
                      margin=dict(t=40, b=40))
    with st.container(border=True):
        st.markdown('<div class="am-card-title">Profils atmosphériques '
                    '0 – 20 km</div>', unsafe_allow_html=True)
        st.caption(f"ΔISA = {disa:+.0f} °C · point courant à {fr(h)} m")
        st.plotly_chart(fig)

    with st.expander("Altitude à partir de la pression"):
        p_query = st.number_input("Pression P [Pa]", 5000.0, 110000.0,
                                  float(props['P']), 100.0)
        h_inv = mod_atm.altitude_from_pressure(p_query, disa)
        st.metric("Altitude h", f"{float(h_inv):,.1f} m")


def page_conv():
    page_head("Module de conversion — TAS / CAS / Mach",
              "Conversions isentropiques entre vitesse vraie, vitesse "
              "calibrée et nombre de Mach, basées sur l'atmosphère ISA.")

    convs = {
        "Mach → TAS": (mod_conv.mach_to_tas, "mach", "vitesse"),
        "TAS → Mach": (mod_conv.tas_to_mach, "vitesse", "mach"),
        "Mach → CAS": (mod_conv.mach_to_cas, "mach", "vitesse"),
        "CAS → Mach": (mod_conv.cas_to_mach, "vitesse", "mach"),
        "TAS → CAS": (mod_conv.tas_to_cas, "vitesse", "vitesse"),
        "CAS → TAS": (mod_conv.cas_to_tas, "vitesse", "vitesse"),
    }

    with st.container(border=True):
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
                val_in = cc1.number_input(f"Vitesse [{unite}]", 0.0, 1200.0,
                                          300.0, 1.0)
                x = val_in * KT if unite == "kt" else val_in
            cc3, cc4 = st.columns(2)
            h = cc3.slider("Altitude h [m]", 0.0, 20000.0, 10000.0, 50.0)
            disa = cc4.slider("ΔISA [°C]", -30.0, 30.0, 0.0, 1.0)

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
        metrics_card("Résultat", items, cols=1)

    hs = np.linspace(0.0, 20000.0, 201)
    ys = np.array([float(f(x, hh, disa)) for hh in hs])
    if kind_out == "vitesse" and unite == "kt":
        ys = ys / KT
    y_label = "Mach" if kind_out == "mach" else f"Vitesse [{unite}]"
    cur = res if kind_out == "mach" else (res / KT if unite == "kt" else res)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ys, y=hs, mode="lines",
                             line=dict(color=NAVY, width=2),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[cur], y=[h], mode="markers",
                             marker=dict(color=RED, size=22, opacity=.15),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[cur], y=[h], mode="markers",
                             marker=dict(color=RED, size=9),
                             showlegend=False))
    fig.update_layout(height=420, template="plotly_white",
                      xaxis_title=y_label, yaxis_title="h [m]",
                      margin=dict(t=40, b=40))
    with st.container(border=True):
        st.markdown(f'<div class="am-card-title">{choix} en fonction de '
                    'l\'altitude</div>', unsafe_allow_html=True)
        st.caption(f"Entrée constante : {val_in:g} "
                   f"{'Mach' if kind_in == 'mach' else unite} · "
                   f"ΔISA = {disa:+.0f} °C")
        st.plotly_chart(fig)


def page_aero():
    page_head("Module aérodynamique — A380",
              "Coefficients de portance, de traînée et de moment construits "
              "sur les données OpenVSP / VSPAERO — aile-fuselage, empennage "
              "horizontal et avion complet.")

    model = load_aero_model()
    grid = model['f_clwb']
    a_min, a_max = float(grid['x_alpha'][0]), float(grid['x_alpha'][-1])
    m_min, m_max = float(grid['y_mach'][0]), float(grid['y_mach'][-1])

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        alpha = c1.slider("Angle d'attaque α [deg]", a_min, a_max, 2.0, 0.1)
        mach = c2.slider("Mach", m_min, m_max, min(0.85, m_max), 0.01)
        dit = c3.slider("Calage empennage δ_it [deg]", -10.0, 10.0, 0.0, 0.5)

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
                metric("CL", f"{cl:.4f}"),
                metric("CD", f"{cd:.5f}"),
                metric("CM", f"{cm:.4f}"),
            ], cols=3, small=True)

    ratios_strip([
        ("Downwash ε", f"{eps:.3f}°"),
        ("α_ht = α − ε + δ_it", f"{alpha - eps + dit:.3f}°"),
        ("Finesse CL_t/CD_t", f"{cl_t / cd_t:.2f}" if cd_t else "—"),
    ])

    tab1, tab2, tab3 = st.tabs(["Coefficients vs α", "Polaire", "Surface 3D"])

    curves = aero_curves(dit, mach)
    alphas = curves['alpha']

    with tab1:
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=("CL", "CD", "CM"))
        for i, key in enumerate(("cl", "cd", "cm"), start=1):
            fig.add_trace(go.Scatter(x=alphas, y=curves[f'{key}_t'],
                                     name="total", legendgroup="t",
                                     showlegend=(i == 1),
                                     line=dict(color=NAVY, width=2)),
                          row=1, col=i)
            fig.add_trace(go.Scatter(x=alphas, y=curves[f'{key}_wb'],
                                     name="WB seul", legendgroup="wb",
                                     showlegend=(i == 1),
                                     line=dict(color="#8B93A1", dash="dash")),
                          row=1, col=i)
            cur = {"cl": rows[2][1], "cd": rows[2][2], "cm": rows[2][3]}[key]
            fig.add_trace(go.Scatter(x=[alpha], y=[cur], mode="markers",
                                     marker=dict(color=RED, size=9),
                                     showlegend=False), row=1, col=i)
        fig.update_xaxes(title_text="α [deg]")
        fig.update_layout(height=420, template="plotly_white",
                          title=f"Coefficients totaux — M = {mach:.2f}, "
                                f"δ_it = {dit:+.1f}°")
        st.plotly_chart(fig)

    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curves['cd_t'], y=curves['cl_t'],
                                 mode="lines", line=dict(color=NAVY, width=2),
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=[rows[2][2]], y=[rows[2][1]],
                                 mode="markers",
                                 marker=dict(color=RED, size=9),
                                 showlegend=False))
        fig.update_layout(height=420, template="plotly_white",
                          xaxis_title="CD_t", yaxis_title="CL_t",
                          title=f"Polaire avion complet — M = {mach:.2f}, "
                                f"δ_it = {dit:+.1f}°")
        st.plotly_chart(fig)

    with tab3:
        coef = st.selectbox("Coefficient", ["CL_t", "CD_t", "CM_t"])
        a_s, m_s, z = aero_surface(coef, dit)
        fig = go.Figure(go.Surface(x=m_s, y=a_s, z=z, colorscale="Viridis"))
        fig.update_layout(height=560, template="plotly_white",
                          scene=dict(xaxis_title="Mach",
                                     yaxis_title="α [deg]",
                                     zaxis_title=coef),
                          title=f"{coef}(α, M) — δ_it = {dit:+.1f}°")
        st.plotly_chart(fig)


def page_prop():
    page_head("Module de propulsion & émissions — Trent 970",
              "Poussée nette et débit carburant par polynômes de surface "
              "calibrés ; indices d'émission OACI généralisés aux conditions "
              "de vol par la méthode Boeing Fuel Flow.")

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        n1 = c1.slider("Fan speed N1 [%]", 60.0, 100.0, 90.0, 0.5)
        mach = c2.slider("Mach", 0.0, 0.85, 0.85, 0.01)
        h = c3.slider("Altitude h [m]", 0.0, 15000.0, 10668.0, 50.0)
        disa = c4.slider("ΔISA [°C]", -30.0, 30.0, 0.0, 1.0)

    fn = float(mod_prop.get_thrust(n1, mach, h, disa))
    ei = mod_prop.get_emission_indices(n1, mach, h, disa)
    wf = float(ei['WF'])
    theta = float(mod_atm.theta(h, disa))
    delta = float(mod_atm.delta(h, disa))

    with st.container(border=True):
        metrics_card("Performances moteur", [
            metric("Poussée nette FN", f"{fn:.4f}", "",
                   f"× 4 moteurs = {4 * fn:.4f}"),
            metric("Débit carburant WF", f"{wf:.4f}", "kg/s",
                   f"{fr(wf * 3600)} kg/h"),
            metric("W_F,C^REF (BFF)", f"{ei['WF_C_REF']:.4f}", "kg/s",
                   "table OACI : 0.261 – 2.738"),
        ], cols=3, small=True)

    with st.container(border=True):
        metrics_card("Indices d'émission généralisés (méthode Boeing "
                     "Fuel Flow)", [
            metric("EI NOx", f"{ei['EI_NOx']:.3f}", "g/kg",
                   f"{ei['EI_NOx'] * wf:.3f} g/s"),
            metric("EI UHC", f"{ei['EI_UHC']:.4f}", "g/kg",
                   f"{ei['EI_UHC'] * wf:.4f} g/s"),
            metric("EI CO", f"{ei['EI_CO']:.3f}", "g/kg",
                   f"{ei['EI_CO'] * wf:.3f} g/s"),
            metric("EI CO₂", f"{ei['EI_CO2']:.2f}", "kg/kg",
                   f"{ei['EI_CO2'] * wf:.3f} kg/s"),
        ], cols=4, small=True)

    ratios_strip([
        ("N1_cor = N1/√θ", f"{n1 / np.sqrt(theta):.2f} %"),
        ("W_F,C = WF/(δ√θ)", f"{wf / (delta * np.sqrt(theta)):.4f} kg/s"),
        ("Humidité spécifique ω", f"{ei['omega']:.6f}"),
    ])

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
        ], cols=5, small=True)

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
                                     showlegend=True), row=1, col=1)
            fig.add_trace(go.Scatter(x=n1s,
                                     y=mod_prop.get_fuel_flow(n1s, M, h, disa),
                                     name=f"M={M:.2f}", legendgroup=f"{M:.2f}",
                                     showlegend=False), row=1, col=2)
        fig.update_xaxes(title_text="N1 [%]")
        fig.update_layout(height=440, template="plotly_white",
                          title=f"h = {h:.0f} m  |  ΔISA = {disa:+.0f} °C")
        st.plotly_chart(fig)

    with tab2:
        machs_3d = np.linspace(0.0, 0.85, 40)
        N1g, Mg = np.meshgrid(n1s, machs_3d)
        fn_surf = mod_prop.get_thrust(N1g, Mg, h, disa)
        wf_surf = mod_prop.get_fuel_flow(N1g, Mg, h, disa)
        cc1, cc2 = st.columns(2)
        for col, (surf, nom) in zip((cc1, cc2),
                                    ((fn_surf, "FN"), (wf_surf, "WF [kg/s]"))):
            fig = go.Figure(go.Surface(x=N1g, y=Mg, z=surf,
                                       colorscale="Viridis"))
            fig.update_layout(height=480, template="plotly_white",
                              scene=dict(xaxis_title="N1 [%]",
                                         yaxis_title="Mach",
                                         zaxis_title=nom),
                              title=nom)
            col.plotly_chart(fig)

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
                                         showlegend=(i == 1)), row=1, col=i)
        fig.update_xaxes(title_text="N1 [%]")
        fig.update_layout(height=440, template="plotly_white",
                          title=f"Indices d'émission — h = {h:.0f} m  |  "
                                f"ΔISA = {disa:+.0f} °C")
        st.plotly_chart(fig)


def page_trim():
    page_head("Module d'équilibrage (Trim)")
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


def apply_sidebar_background():
    """Image de flux d'air en fond de sidebar, texte passé en blanc.

    CSS volontairement confiné à [data-testid="stSidebar"] : on ne touche
    ni aux polices ni à la visibilité d'éléments (cf. mésaventure des
    icônes Material avec le CSS global).
    """
    if not SIDEBAR_IMG.exists():
        return
    b64 = _img_b64(str(SIDEBAR_IMG), SIDEBAR_IMG.stat().st_mtime)
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
        background: rgba(255, 255, 255, .16);
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
