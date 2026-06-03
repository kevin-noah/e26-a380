# Performances Airbus A380 — MGA803

Modélisation des performances de l'Airbus A380 dans le cadre du cours MGA803 (ÉTS).

## Lancement

```bash
python main.py          # mode interactif
python main.py atm isa --h 10000 --disa 0   # mode direct
python main.py conv mach2tas --m 0.82 --h 10000
```

## Modules

| # | Clé | Module | Statut |
|---|-----|--------|--------|
| 1 | `atm` | Atmosphère ISA | ✅ |
| 2 | `conv` | Conversion TAS / CAS / Mach | ✅ |
| 3 | `aero` | Aérodynamique (OpenVSP) | ✅ |
| 4 | `prop` | Propulsion | ✅ *(émissions OACI à venir)* |
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

### 4. Module Propulsion (`prop`)

Polynôme de surface calibré (degré 5 en N1_cor, degré 3 en Mach) avec corrections δ/θ.

```
PROP> thrust --n1 85 --mach 0.82 --h 10000
PROP> wf     --n1 85 --mach 0.82 --h 10000
PROP> all    --n1 85 --mach 0.82 --h 10000 --disa 10
PROP> plot   --h 10000
```

- Poussée : `FN = f̄_th(N1/√θ, M) × δ`
- Débit : `WF = f̄_wf(N1/√θ, M) × δ√θ`

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
```
