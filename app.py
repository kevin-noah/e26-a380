"""
Application Streamlit — Performances Airbus A380 (MGA803)

Interface web interactive regroupant les modules du projet :
atmosphère ISA, conversions de vitesses, aérodynamique (OpenVSP),
propulsion et émissions OACI.

Lancement : streamlit run app.py
"""

import base64
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import atmosphere as mod_atm
import conversion as mod_conv
import aerodynamics as mod_aero
import propulsion as mod_prop

KT = 0.514444   # 1 kt en m/s

ASSETS = Path(__file__).parent / "assets"
HERO_VIDEO = ASSETS / "video-accueil.mp4"
SIDEBAR_IMG = ASSETS / "sidebar-airflow.jpg"

st.set_page_config(page_title="A380 — MGA803", page_icon="✈️", layout="wide")


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


def page_accueil():
    if HERO_VIDEO.exists():
        b64 = _video_b64(str(HERO_VIDEO), HERO_VIDEO.stat().st_mtime)
        st.iframe(_HERO_HTML.format(b64=b64), height=348)
    else:
        st.title("✈️ Performances Airbus A380")
        st.caption("MGA803 — Analyse et optimisation des performances "
                   "des avions · ÉTS, É2026")
    st.markdown(
        """
        Cette application regroupe les modules de modélisation développés
        dans le cadre du projet :

        | Module | Contenu |
        |---|---|
        | **Atmosphère** | Modèle ISA : T, P, ρ, a et ratios θ, δ, σ (0 – 20 000 m, ΔISA) |
        | **Conversion** | Conversions TAS / CAS / Mach (formules isentropiques) |
        | **Aérodynamique** | Coefficients CL, CD, CM (données OpenVSP / VSPAERO), downwash, totaux avion |
        | **Propulsion & Émissions** | Poussée FN, débit carburant WF (Trent 970) et émissions OACI (méthode Boeing Fuel Flow) |
        | **Équilibrage (Trim)** | À venir |

        Sélectionnez un module dans la barre latérale.
        """
    )


def page_atm():
    st.header("🌡️ Module Atmosphérique — ISA")

    c1, c2 = st.columns(2)
    h = c1.slider("Altitude h [m]", 0.0, 20000.0, 10000.0, 50.0)
    disa = c2.slider("ΔISA [°C]", -30.0, 30.0, 0.0, 1.0)

    props = mod_atm.atmosphere(h, disa)

    c = st.columns(4)
    c[0].metric("Température T", f"{props['T']:.2f} K",
                f"{props['T'] - 273.15:.2f} °C", delta_color="off")
    c[1].metric("Pression P", f"{props['P']:,.0f} Pa")
    c[2].metric("Masse volumique ρ", f"{props['rho']:.4f} kg/m³")
    c[3].metric("Vitesse du son a", f"{props['a']:.2f} m/s")

    c = st.columns(4)
    c[0].metric("θ = T/T₀", f"{props['theta']:.6f}")
    c[1].metric("δ = P/P₀", f"{props['delta']:.6f}")
    c[2].metric("σ = ρ/ρ₀", f"{props['sigma']:.6f}")

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
                                 line=dict(color="#1f77b4"),
                                 showlegend=False), row=1, col=i)
        fig.add_trace(go.Scatter(x=[cur], y=[h], mode="markers",
                                 marker=dict(color="crimson", size=9),
                                 showlegend=False), row=1, col=i)
    fig.update_yaxes(title_text="h [m]", row=1, col=1)
    fig.update_layout(height=420, template="plotly_white",
                      title=f"Profils atmosphériques — ΔISA = {disa:+.0f} °C")
    st.plotly_chart(fig)

    with st.expander("Altitude à partir de la pression"):
        p_query = st.number_input("Pression P [Pa]", 5000.0, 110000.0,
                                  float(props['P']), 100.0)
        h_inv = mod_atm.altitude_from_pressure(p_query, disa)
        st.metric("Altitude h", f"{float(h_inv):,.1f} m")


def page_conv():
    st.header("🔄 Module de Conversion — TAS / CAS / Mach")

    convs = {
        "Mach → TAS": (mod_conv.mach_to_tas, "mach", "vitesse"),
        "TAS → Mach": (mod_conv.tas_to_mach, "vitesse", "mach"),
        "Mach → CAS": (mod_conv.mach_to_cas, "mach", "vitesse"),
        "CAS → Mach": (mod_conv.cas_to_mach, "vitesse", "mach"),
        "TAS → CAS": (mod_conv.tas_to_cas, "vitesse", "vitesse"),
        "CAS → TAS": (mod_conv.cas_to_tas, "vitesse", "vitesse"),
    }

    c1, c2 = st.columns([1, 2])
    choix = c1.radio("Conversion", list(convs))
    f, kind_in, kind_out = convs[choix]

    with c2:
        cc1, cc2 = st.columns(2)
        if kind_in == "mach":
            val_in = cc1.number_input("Mach", 0.0, 1.0, 0.85, 0.01)
            x = val_in
            unite = cc2.radio("Unité de sortie", ["kt", "m/s"], horizontal=True)
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
    if kind_out == "mach":
        st.metric(choix, f"M = {res:.4f}")
    else:
        res_aff = res / KT if unite == "kt" else res
        st.metric(choix, f"{res_aff:.2f} {unite}",
                  f"{res:.2f} m/s" if unite == "kt" else f"{res / KT:.2f} kt",
                  delta_color="off")

    hs = np.linspace(0.0, 20000.0, 201)
    ys = np.array([float(f(x, hh, disa)) for hh in hs])
    if kind_out == "vitesse" and unite == "kt":
        ys = ys / KT
    y_label = "Mach" if kind_out == "mach" else f"Vitesse [{unite}]"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ys, y=hs, mode="lines",
                             line=dict(color="#1f77b4"), showlegend=False))
    fig.add_trace(go.Scatter(x=[res if kind_out == "mach" else
                                (res / KT if unite == "kt" else res)],
                             y=[h], mode="markers",
                             marker=dict(color="crimson", size=9),
                             showlegend=False))
    fig.update_layout(height=420, template="plotly_white",
                      xaxis_title=y_label, yaxis_title="h [m]",
                      title=f"{choix} en fonction de l'altitude "
                            f"(entrée constante : "
                            f"{val_in:g} {'Mach' if kind_in == 'mach' else unite})")
    st.plotly_chart(fig)


def page_aero():
    st.header("🛩️ Module Aérodynamique — A380 (OpenVSP)")

    model = load_aero_model()
    grid = model['f_clwb']
    a_min, a_max = float(grid['x_alpha'][0]), float(grid['x_alpha'][-1])
    m_min, m_max = float(grid['y_mach'][0]), float(grid['y_mach'][-1])

    c1, c2, c3 = st.columns(3)
    alpha = c1.slider("Angle d'attaque α [deg]", a_min, a_max, 2.0, 0.1)
    mach = c2.slider("Mach", m_min, m_max, min(0.85, m_max), 0.01)
    dit = c3.slider("Calage empennage δ_it [deg]", -10.0, 10.0, 0.0, 0.5)

    eps = float(mod_aero.f_downwash(alpha))
    st.caption(f"Downwash : ε = {eps:.3f}°   →   "
               f"α_ht = α − ε + δ_it = {alpha - eps + dit:.3f}°")

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
    for nom, cl, cd, cm in rows:
        c = st.columns([1.4, 1, 1, 1])
        c[0].markdown(f"**{nom}**")
        c[1].metric("CL", f"{cl:.4f}")
        c[2].metric("CD", f"{cd:.5f}")
        c[3].metric("CM", f"{cm:.4f}")

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
                                     line=dict(color="#1f77b4")), row=1, col=i)
            fig.add_trace(go.Scatter(x=alphas, y=curves[f'{key}_wb'],
                                     name="WB seul", legendgroup="wb",
                                     showlegend=(i == 1),
                                     line=dict(color="#888", dash="dash")),
                          row=1, col=i)
            cur = {"cl": rows[2][1], "cd": rows[2][2], "cm": rows[2][3]}[key]
            fig.add_trace(go.Scatter(x=[alpha], y=[cur], mode="markers",
                                     marker=dict(color="crimson", size=9),
                                     showlegend=False), row=1, col=i)
        fig.update_xaxes(title_text="α [deg]")
        fig.update_layout(height=420, template="plotly_white",
                          title=f"Coefficients totaux — M = {mach:.2f}, "
                                f"δ_it = {dit:+.1f}°")
        st.plotly_chart(fig)

    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curves['cd_t'], y=curves['cl_t'],
                                 mode="lines", line=dict(color="#1f77b4"),
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=[rows[2][2]], y=[rows[2][1]],
                                 mode="markers",
                                 marker=dict(color="crimson", size=9),
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
    st.header("🚀 Module de Propulsion & Émissions — Trent 970")

    c1, c2, c3, c4 = st.columns(4)
    n1 = c1.slider("Fan speed N1 [%]", 60.0, 100.0, 90.0, 0.5)
    mach = c2.slider("Mach", 0.0, 0.85, 0.85, 0.01)
    h = c3.slider("Altitude h [m]", 0.0, 15000.0, 10668.0, 50.0)
    disa = c4.slider("ΔISA [°C]", -30.0, 30.0, 0.0, 1.0)

    fn = float(mod_prop.get_thrust(n1, mach, h, disa))
    ei = mod_prop.get_emission_indices(n1, mach, h, disa)

    st.subheader("Performances moteur")
    c = st.columns(3)
    c[0].metric("Poussée nette FN", f"{fn:.4f}")
    c[1].metric("Débit carburant WF", f"{ei['WF']:.4f} kg/s")
    c[2].metric("W_F,C^REF (BFF)", f"{ei['WF_C_REF']:.4f} kg/s")

    st.subheader("Indices d'émission généralisés (méthode Boeing Fuel Flow)")
    c = st.columns(4)
    c[0].metric("EI NOx", f"{ei['EI_NOx']:.3f} g/kg")
    c[1].metric("EI UHC", f"{ei['EI_UHC']:.4f} g/kg")
    c[2].metric("EI CO", f"{ei['EI_CO']:.3f} g/kg")
    c[3].metric("EI CO₂", f"{ei['EI_CO2']:.2f} kg/kg")
    st.caption(f"Facteur spécifique d'humidité : ω = {ei['omega']:.6f} "
               f"(RH = 0.80)")

    with st.expander("Masses de polluants émises"):
        cc1, cc2 = st.columns(2)
        t = cc1.number_input("Durée [s]", 1.0, 36000.0, 3600.0, 60.0)
        n_mot = cc2.radio("Moteurs", [1, 4], horizontal=True)
        em = mod_prop.get_emissions(n1, mach, h, disa, duration=t)
        c = st.columns(5)
        c[0].metric("Carburant brûlé", f"{em['fuel_burn'] * n_mot:,.1f} kg")
        c[1].metric("NOx", f"{em['m_NOx'] * n_mot / 1000:,.2f} kg")
        c[2].metric("UHC", f"{em['m_UHC'] * n_mot / 1000:,.3f} kg")
        c[3].metric("CO", f"{em['m_CO'] * n_mot / 1000:,.2f} kg")
        c[4].metric("CO₂", f"{em['m_CO2'] * n_mot:,.0f} kg")

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
    st.header("⚖️ Module d'Équilibrage (Trim)")
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
    </style>
    """, unsafe_allow_html=True)


apply_sidebar_background()
st.sidebar.title("✈️ A380 — MGA803")
choix_page = st.sidebar.radio("Module", list(PAGES))
st.sidebar.caption("Analyse et optimisation des performances des avions — "
                   "ÉTS, É2026")

PAGES[choix_page]()
