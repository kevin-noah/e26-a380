# Performances Airbus A380 — MGA803

Modélisation des performances de l'Airbus A380 dans le cadre du cours MGA803 (ÉTS).

## Lancement

```bash
python main.py          # mode interactif
python main.py atm isa --h 10000 --disa 0   # mode direct
python main.py conv mach2tas --m 0.82 --h 10000

streamlit run app.py    # interface web interactive
```

## Modules

| # | Clé | Module | Statut |
|---|-----|--------|--------|
| 1 | `atm` | Atmosphère ISA | ✅ |
| 2 | `conv` | Conversion TAS / CAS / Mach | ✅ |
| 3 | `aero` | Aérodynamique (OpenVSP) | ✅ |
| 4 | `prop` | Propulsion & émissions OACI | ✅ |
| 5 | `trim` | Équilibrage | 🚧 en développement |

---

### 1. Module Atmosphérique (`atm`)

Atmosphère ISA sur 0–20 000 m (troposphère + basse stratosphère).

```
ATM> isa --h 10000
ATM> isa --h 10000 --disa 15
ATM> isa --p 50000
```

Sorties : T, P, ρ, a, θ, δ, σ.

---

### 2. Module Conversion (`conv`)

Conversions entre TAS, CAS et Mach (formules isentropiques subsoniques).

```
CONV> mach2tas --m 0.82 --h 10000
CONV> tas2cas  --v 250  --h 10000        # vitesse en kt par défaut
CONV> cas2mach --v 480  --h 8000 --unite kt
```

---

### 3. Module Aérodynamique (`aero`)

Tables d'interpolation 2D (α, Mach) construites depuis les fichiers VSPAERO (OpenVSP).
Coefficients pour l'aile+fuselage (WB) et l'empennage horizontal (HT).

```
AERO> load
AERO> all --alpha 3 --mach 0.82
AERO> cmt --alpha 3 --mach 0.82 --delta-it 2
AERO> plot_total --delta-it 2
AERO> geom
```

Coefficients disponibles : `CL`, `CD`, `Cm` pour WB, HT et avion complet.

**Géométrie de référence A380**

| Paramètre | Valeur |
|-----------|--------|
| S_WB | 859 m² |
| c_WB | 11 m |
| S_HT | 205 m² |
| c_HT | 6.77 m |
| x_HT | 32 m |
| z_HT | 1.24 m |

---

### 4. Module Propulsion & Émissions (`prop`)

Polynôme de surface calibré (degré 5 en N1_cor, degré 3 en Mach) avec corrections δ/θ.

```
PROP> thrust --n1 85 --mach 0.82 --h 10000
PROP> wf     --n1 85 --mach 0.82 --h 10000
PROP> ei     --n1 85 --mach 0.82 --h 10000
PROP> emis   --n1 85 --mach 0.82 --h 10000 --t 3600
PROP> all    --n1 85 --mach 0.82 --h 10000 --disa 10
PROP> plot   --h 10000
PROP> plot_emis --h 10000
```

- Poussée : `FN = f̄_th(N1/√θ, M) × δ`
- Débit : `WF = f̄_wf(N1/√θ, M) × δ√θ`

**Émissions OACI — méthode Boeing Fuel Flow (BFF)**

Indices d'émission NOx, UHC, CO généralisés aux conditions de vol à partir
des données de référence du Trent 970B-84 (Banque de données OACI) :

1. Débit corrigé : `W_F,C = WF / (δ√θ)`
2. Débit de référence équivalent : `W_F,C^REF = (1+0.2M²)·θ^4.3·W_F,C` (x=1, y=0.5)
3. Interpolation ln-ln des indices de référence sur le cycle LTO
4. Décorrélation : `EI_UHC/CO = EI^REF·θ^3.3/δ^1.02`,
   `EI_NOx = EI^REF·√(δ^1.02/θ^3.3)·exp[−19(ω−0.00634)]`

CO₂ : modèle proportionnel `EI_CO2 = 3.16 kg/kg`.
Masses émises : `Δm_i = EI_i × ΔF_B`.

---

### Interface web (Streamlit)

`app.py` regroupe tous les modules dans une interface interactive
(sliders, graphiques Plotly 2D/3D) pour la présentation :

```bash
streamlit run app.py
```

---

## Données

```
data/
  Avion_WB_VSPGeom.history   — coefficients aile + fuselage (VSPAERO)
  Avion_HT_VSPGeom.history   — coefficients empennage horizontal (VSPAERO)
  Avion_WBT_VSPGeom.vspgeom  — géométrie 3D OpenVSP
```

## Auteurs

- Rodrigue Fosing
- Valentin Durand
- Kevin Noah

## Dépendances

```
numpy
scipy
pandas
matplotlib
streamlit   # interface web
plotly      # graphiques interactifs
```
