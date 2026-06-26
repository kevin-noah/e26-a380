"""
Modélisation des performances de l'Airbus A380 — MGA803

Usage :
    python main.py                                      Mode interactif
    python main.py atm isa --h 3000 --disa 20           Mode direct
    python main.py conv mach2tas --m 0.82 --h 10000     Mode direct
"""

import argparse
import os
import shlex   # découpage d'une chaîne en tokens comme le ferait un shell
import sys
import numpy as np

import atmosphere    as mod_atm
import conversion    as mod_conv
import aerodynamics  as mod_aero
import propulsion    as mod_prop
import trim          as mod_trim
import performance   as mod_perf

# ---------------------------------------------------------------------------
# Registre des modules — chaque entrée porte son module et son statut
# ---------------------------------------------------------------------------
# registre des modules
MODULES = [
    {"id": 1, "cle": "atm",  "module": mod_atm,  "dispo": True},
    {"id": 2, "cle": "conv", "module": mod_conv,  "dispo": True},
    {"id": 3, "cle": "aero", "module": mod_aero,  "dispo": True},
    {"id": 4, "cle": "prop", "module": mod_prop,  "dispo": True},
    {"id": 5, "cle": "trim", "module": mod_trim,  "dispo": True},
    {"id": 6, "cle": "perf", "module": mod_perf,  "dispo": True},
]


# ---------------------------------------------------------------------------
# Menu principal
# ---------------------------------------------------------------------------

def print_main_menu():
    print()
    print("=" * 60)
    print("       Performances Airbus A380 — MGA803")
    print("=" * 60)
    print()
    for m in MODULES:
        # Indique visuellement les modules non encore disponibles
        if not m["dispo"]:
            statut = "  [en développement]"
        elif m.get("note"):
            statut = f"  [{m['note']}]"
        else:
            statut = ""
        print(f"  {m['id']}  {m['module'].NOM:<30}{statut}")
        print(f"     {m['module'].DESCRIPTION}")
        print()
    print("  Tapez le numéro ou la clé du module  (ex: 1  ou  atm)")
    print("  quit   Quitter")
    print("=" * 60)
    print()


# ============================= MODULE ATMOSPHÉRIQUE ============================

def print_atm_menu():
    print()
    print("─" * 50)
    print(f"  {mod_atm.NOM}")
    print("─" * 50)
    print("  isa --h <altitude> [--disa <val>]")
    print("      Propriétés à une altitude [m]  (0 ≤ h ≤ 20 000 m)")
    print()
    print("  isa --p <pression> [--disa <val>]")
    print("      Propriétés à une pression [Pa]")
    print()
    print("  help   Afficher cette aide")
    print("  back   Retour au menu principal")
    print("─" * 50)
    print()


def print_atm_single(h, delta_isa):
    """Affiche toutes les propriétés atmosphériques pour une altitude donnée."""
    props  = mod_atm.atmosphere(h, delta_isa)
    couche = "Troposphère" if h <= mod_atm.H_TROPO else "Stratosphère basse"
    print()
    print("=" * 46)
    print(f"  Atmosphère ISA  |  ΔISA = {delta_isa:+.1f} °C")
    print(f"  Couche : {couche}")
    print("=" * 46)
    # Propriétés dimensionnelles
    print(f"  Altitude             h  = {h:>10.2f}  m")
    print(f"  Température          T  = {props['T']:>10.3f}  K")
    print(f"  Pression             P  = {props['P']:>10.2f}  Pa")
    print(f"  Masse volumique      ρ  = {props['rho']:>10.5f}  kg/m³")
    print(f"  Vitesse du son       a  = {props['a']:>10.3f}  m/s")
    print("-" * 46)
    # Ratios sans dimension (normalisés par les valeurs MSL)
    print(f"  Ratio température    θ  = {props['theta']:>10.6f}")
    print(f"  Ratio pression       δ  = {props['delta']:>10.6f}")
    print(f"  Ratio masse vol.     σ  = {props['sigma']:>10.6f}")
    print("=" * 46)
    print()


# Sous-classe d'ArgumentParser qui lève ValueError au lieu d'appeler sys.exit()
# — indispensable en mode interactif pour ne pas fermer le programme sur une erreur
class _SilentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def _build_atm_parser():
    """Construit le parser de la sous-commande 'isa' (--h et --p mutuellement exclusifs)."""
    parser = _SilentParser(prog="")
    sub    = parser.add_subparsers(dest="cmd")
    sub.required = True

    p_isa = sub.add_parser("isa")
    # L'utilisateur fournit soit l'altitude, soit la pression — jamais les deux
    grp = p_isa.add_mutually_exclusive_group(required=True)
    grp.add_argument("--h",    type=float, metavar="ALTITUDE")
    grp.add_argument("--p",    type=float, metavar="PRESSION")
    p_isa.add_argument("--disa", type=float, default=0.0)

    return parser


def loop_atm():
    """Boucle interactive du module atmosphérique."""
    print_atm_menu()
    parser = _build_atm_parser()

    while True:
        try:
            ligne = input("  ATM> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D ou Ctrl+C : sortie propre vers le menu principal
            print()
            break

        if not ligne:
            continue
        if ligne.lower() in ("back", "menu", "b"):
            break
        if ligne.lower() in ("help", "aide", "?"):
            print_atm_menu()
            continue

        try:
            # shlex.split respecte les guillemets comme un vrai shell
            args = parser.parse_args(shlex.split(ligne))

            if args.cmd == "isa":
                if args.h is not None:
                    # Validation de la plage d'altitude (modèle limité à 20 000 m)
                    if not (0 <= args.h <= 20000):
                        raise ValueError("L'altitude --h doit être comprise entre 0 et 20 000 m.")
                    h = args.h
                else:
                    # Inversion P → h par les formules analytiques du module atm
                    h = mod_atm.altitude_from_pressure(args.p, args.disa)
                    if h > 20000:
                        raise ValueError(f"La pression correspond à h = {h:.0f} m > 20 000 m.")
                print_atm_single(h, args.disa)

        except ValueError as e:
            print(f"  Erreur : {e}\n")
        except SystemExit:
            # argparse appelle sys.exit() pour --help ; on l'intercepte
            pass


# ============================== MODULE CONVERSION ==============================

_KT = 1 / 0.514444   # facteur de conversion m/s → kt (1 kt = 0.514444 m/s)


def print_conv_menu():
    print()
    print("─" * 58)
    print(f"  {mod_conv.NOM}")
    print("─" * 58)
    print("  Toutes les commandes nécessitent --h [m] et acceptent --disa [°C].")
    print("  Les vitesses (TAS, CAS) s'expriment en kt par défaut (--unite ms).")
    print()
    print("  mach2tas --m <M>  --h <h> [--disa <v>] [--unite kt|ms]")
    print("  tas2mach --v <v>  --h <h> [--disa <v>] [--unite kt|ms]")
    print("  mach2cas --m <M>  --h <h> [--disa <v>] [--unite kt|ms]")
    print("  cas2mach --v <v>  --h <h> [--disa <v>] [--unite kt|ms]")
    print("  tas2cas  --v <v>  --h <h> [--disa <v>] [--unite kt|ms]")
    print("  cas2tas  --v <v>  --h <h> [--disa <v>] [--unite kt|ms]")
    print()
    print("  help   Afficher cette aide")
    print("  back   Retour au menu principal")
    print("─" * 58)
    print()


def _print_conv_aero(label_in, val_in, label_out, val_out, mach, h, delta_isa):
    """Affiche le résultat d'une conversion aéronautique avec les conditions atmosph."""
    # Conditions atmosphériques à l'altitude demandée
    theta = mod_atm.theta(h, delta_isa)
    delta = mod_atm.delta(h, delta_isa)
    a     = mod_atm.A0 * np.sqrt(theta)  # vitesse du son locale = a0·√θ

    # Formatage d'une vitesse en kt et m/s simultanément
    def fmt_spd(v_ms):
        return f"{v_ms * _KT:>9.3f} kt   ({v_ms:>9.3f} m/s)"

    print()
    print("=" * 56)
    # En-tête : conditions ISA utilisées pour la conversion
    print(f"  h = {h:.0f} m  |  ΔISA = {delta_isa:+.1f} °C  |  "
          f"θ = {theta:.4f}  |  a = {a:.3f} m/s")
    print(f"  δ = {delta:.4f}")
    print("─" * 56)

    # Affichage de la valeur d'entrée
    if label_in == "Mach":
        print(f"  {'Mach':<6}  =  {val_in:.6f}")
    else:
        print(f"  {label_in:<6}  =  {fmt_spd(val_in)}")

    # Affichage de la valeur de sortie
    if label_out == "Mach":
        print(f"  {'Mach':<6}  =  {val_out:.6f}")
    else:
        print(f"  {label_out:<6}  =  {fmt_spd(val_out)}")

    # Affiche le Mach intermédiaire si la conversion est TAS↔CAS
    if label_in != "Mach" and label_out != "Mach":
        print(f"  {'Mach':<6}  =  {mach:.6f}")

    print("=" * 56)
    print()


def _build_conv_parser():
    """Construit le parser pour les 6 commandes de conversion de vitesse."""
    parser = _SilentParser(prog="")
    sub    = parser.add_subparsers(dest="cmd")
    sub.required = True

    # Commandes avec une vitesse (TAS ou CAS) en entrée
    for cmd in ("tas2cas", "cas2tas", "cas2mach", "tas2mach"):
        p = sub.add_parser(cmd)
        p.add_argument("--v",     type=float, required=True, metavar="VITESSE")
        p.add_argument("--h",     type=float, required=True, metavar="ALTITUDE")
        p.add_argument("--disa",  type=float, default=0.0,   metavar="ΔISA")
        p.add_argument("--unite", type=str,   default="kt",  choices=["kt", "ms"])

    # Commandes avec un nombre de Mach en entrée
    for cmd in ("mach2cas", "mach2tas"):
        p = sub.add_parser(cmd)
        p.add_argument("--m",     type=float, required=True, metavar="MACH")
        p.add_argument("--h",     type=float, required=True, metavar="ALTITUDE")
        p.add_argument("--disa",  type=float, default=0.0,   metavar="ΔISA")
        p.add_argument("--unite", type=str,   default="kt",  choices=["kt", "ms"])

    return parser


def _to_ms(v, unite):
    """Convertit une vitesse vers m/s selon l'unité saisie (kt ou ms)."""
    return v * 0.514444 if unite == "kt" else v


def loop_conv():
    """Boucle interactive du module de conversion de vitesses."""
    print_conv_menu()
    parser = _build_conv_parser()

    while True:
        try:
            ligne = input("  CONV> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not ligne:
            continue
        if ligne.lower() in ("back", "menu", "b"):
            break
        if ligne.lower() in ("help", "aide", "?"):
            print_conv_menu()
            continue

        try:
            args = parser.parse_args(shlex.split(ligne))

            # Dispatch vers la fonction de conversion correspondante
            if args.cmd == "mach2tas":
                tas = mod_conv.mach_to_tas(args.m, args.h, args.disa)
                _print_conv_aero("Mach", args.m, "TAS", tas, args.m, args.h, args.disa)

            elif args.cmd == "tas2mach":
                tas_ms = _to_ms(args.v, args.unite)
                M = mod_conv.tas_to_mach(tas_ms, args.h, args.disa)
                _print_conv_aero("TAS", tas_ms, "Mach", M, M, args.h, args.disa)

            elif args.cmd == "mach2cas":
                cas = mod_conv.mach_to_cas(args.m, args.h, args.disa)
                _print_conv_aero("Mach", args.m, "CAS", cas, args.m, args.h, args.disa)

            elif args.cmd == "cas2mach":
                cas_ms = _to_ms(args.v, args.unite)
                M = mod_conv.cas_to_mach(cas_ms, args.h, args.disa)
                _print_conv_aero("CAS", cas_ms, "Mach", M, M, args.h, args.disa)

            elif args.cmd == "tas2cas":
                tas_ms = _to_ms(args.v, args.unite)
                cas_ms = mod_conv.tas_to_cas(tas_ms, args.h, args.disa)
                # Mach intermédiaire recalculé pour l'affichage
                M = mod_conv.tas_to_mach(tas_ms, args.h, args.disa)
                _print_conv_aero("TAS", tas_ms, "CAS", cas_ms, M, args.h, args.disa)

            elif args.cmd == "cas2tas":
                cas_ms = _to_ms(args.v, args.unite)
                tas_ms = mod_conv.cas_to_tas(cas_ms, args.h, args.disa)
                # Mach intermédiaire recalculé pour l'affichage
                M = mod_conv.cas_to_mach(cas_ms, args.h, args.disa)
                _print_conv_aero("CAS", cas_ms, "TAS", tas_ms, M, args.h, args.disa)

        except ValueError as e:
            print(f"  Erreur : {e}\n")
        except SystemExit:
            pass


# ============================== MODULE AÉRODYNAMIQUE ==========================

def print_aero_menu():
    print()
    print("─" * 62)
    print(f"  {mod_aero.NOM}")
    print("─" * 62)
    print("  load  [--wb <fichier>] [--ht <fichier>]")
    print("        Charge le modèle (fichiers par défaut si omis)")
    print()
    print("  clwb  --alpha <deg> --mach <M>                  CL aile + fuselage")
    print("  cdwb  --alpha <deg> --mach <M>                  CD aile + fuselage")
    print("  cmwb  --alpha <deg> --mach <M>                  Cm aile + fuselage")
    print("  cl_ht --alpha <deg> --mach <M> [--delta-it <d>] CL empennage")
    print("  cd_ht --alpha <deg> --mach <M> [--delta-it <d>] CD empennage")
    print("  cm_ht --alpha <deg> --mach <M> [--delta-it <d>] Cm empennage")
    print("  clt   --alpha <deg> --mach <M> [--delta-it <d>] CL total avion")
    print("  cdt   --alpha <deg> --mach <M> [--delta-it <d>] CD total avion")
    print("  cmt   --alpha <deg> --mach <M> [--delta-it <d>] CM total avion")
    print("  all   --alpha <deg> --mach <M> [--delta-it <d>] Tous les coefficients")
    print()
    print("  downwash --alpha <deg> [--delta-it <deg>]   Angle de downwash ε [deg]")
    print()
    print("  plot              Graphiques 2D et 3D des coefficients WB et HT")
    print("  plot_total [--delta-it <d>]   Graphiques 2D et 3D de CL_t, CD_t, CM_t")
    print("  geom [--file <fichier.vspgeom>]  Géométrie 3D OpenVSP")
    print("  info              Plages α et Mach du modèle chargé")
    print()
    print("  help  Afficher cette aide")
    print("  back  Retour au menu principal")
    print("─" * 62)
    print()


def _build_aero_parser():
    parser = _SilentParser(prog="")
    sub    = parser.add_subparsers(dest="cmd")
    sub.required = True

    p = sub.add_parser("load")
    p.add_argument("--wb",   type=str, default=None)
    p.add_argument("--ht",   type=str, default=None)

    for cmd in ("clwb", "cdwb", "cmwb"):
        p = sub.add_parser(cmd)
        p.add_argument("--alpha", type=float, required=True)
        p.add_argument("--mach",  type=float, required=True)

    for cmd in ("cl_ht", "cd_ht", "cm_ht"):
        p = sub.add_parser(cmd)
        p.add_argument("--alpha",    type=float, required=True)
        p.add_argument("--mach",     type=float, required=True)
        p.add_argument("--delta-it", type=float, default=0.0,
                       dest="delta_it", metavar="DEG")

    for cmd in ("clt", "cdt", "cmt", "all"):
        p = sub.add_parser(cmd)
        p.add_argument("--alpha",    type=float, required=True)
        p.add_argument("--mach",     type=float, required=True)
        p.add_argument("--delta-it", type=float, default=0.0,
                       dest="delta_it", metavar="DEG")

    p = sub.add_parser("downwash")
    p.add_argument("--alpha",    type=float, required=True)
    p.add_argument("--delta-it", type=float, default=0.0,
                   dest="delta_it", metavar="DEG",
                   help="calage de l'empennage δit [deg] (défaut : 0)")

    sub.add_parser("plot")

    p = sub.add_parser("plot_total")
    p.add_argument("--delta-it", type=float, default=0.0,
                   dest="delta_it", metavar="DEG")

    sub.add_parser("info")

    p = sub.add_parser("geom")
    p.add_argument("--file", type=str, default=None)

    return parser


def _print_aero_result(label, value, alpha, mach):
    print()
    print("=" * 46)
    print(f"  α = {alpha:.2f} °    Mach = {mach:.4f}")
    print("─" * 46)
    print(f"  {label:<10} = {value:>14.8f}")
    print("=" * 46)
    print()


def _print_aero_all(model, alpha, mach, delta_it=0.0):
    cl_wb = mod_aero.get_cl_wb(model, alpha, mach)
    cd_wb = mod_aero.get_cd_wb(model, alpha, mach)
    cm_wb = mod_aero.get_cm_wb(model, alpha, mach)
    cl_ht = mod_aero.get_cl_ht(model, alpha, mach, delta_it=delta_it)
    cd_ht = mod_aero.get_cd_ht(model, alpha, mach, delta_it=delta_it)
    cm_ht = mod_aero.get_cm_ht(model, alpha, mach, delta_it=delta_it)
    cl_t  = mod_aero.get_cl_total(model, alpha, mach, delta_it=delta_it)
    cd_t  = mod_aero.get_cd_total(model, alpha, mach, delta_it=delta_it)
    cm_t  = mod_aero.get_cm_total(model, alpha, mach, delta_it=delta_it)
    print()
    print("=" * 46)
    print(f"  α = {alpha:.2f} °    Mach = {mach:.4f}    δit = {delta_it:.2f} °")
    print("─" * 46)
    print(f"  {'CL_wb':<10} = {cl_wb:>14.8f}")
    print(f"  {'CD_wb':<10} = {cd_wb:>14.8f}")
    print(f"  {'Cm_wb':<10} = {cm_wb:>14.8f}")
    print(f"  {'CL_ht':<10} = {cl_ht:>14.8f}")
    print(f"  {'CD_ht':<10} = {cd_ht:>14.8f}")
    print(f"  {'Cm_ht':<10} = {cm_ht:>14.8f}")
    print("─" * 46)
    print(f"  {'CL_t':<10} = {cl_t:>14.8f}")
    print(f"  {'CD_t':<10} = {cd_t:>14.8f}")
    print(f"  {'CM_t':<10} = {cm_t:>14.8f}")
    print("=" * 46)
    print()


def loop_aero():
    """Boucle interactive du module aérodynamique."""
    print_aero_menu()
    parser  = _build_aero_parser()
    _model  = [None]   # conteneur mutable pour conserver le modèle entre les commandes

    while True:
        try:
            ligne = input("  AERO> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not ligne:
            continue
        if ligne.lower() in ("back", "menu", "b"):
            break
        if ligne.lower() in ("help", "aide", "?"):
            print_aero_menu()
            continue

        try:
            args = parser.parse_args(shlex.split(ligne))

            if args.cmd == "load":
                wb = os.path.expanduser(args.wb) if args.wb else mod_aero.DEFAULT_FILE_WB
                ht = os.path.expanduser(args.ht) if args.ht else mod_aero.DEFAULT_FILE_HT
                print(f"  Chargement de {os.path.basename(wb)} et {os.path.basename(ht)} ...")
                _model[0] = mod_aero.build_aero_model(wb, ht)
                m = _model[0]
                print(f"  Modèle chargé.")
                print(f"    WB  α : {m['f_clwb']['x_alpha'][0]:.1f}° → "
                      f"{m['f_clwb']['x_alpha'][-1]:.1f}°  "
                      f"({len(m['f_clwb']['x_alpha'])} points)")
                print(f"    WB  M : {m['f_clwb']['y_mach'][0]:.3f} → "
                      f"{m['f_clwb']['y_mach'][-1]:.3f}  "
                      f"({len(m['f_clwb']['y_mach'])} points)")
                print(f"    HT  α : {m['f_clht']['x_alpha'][0]:.1f}° → "
                      f"{m['f_clht']['x_alpha'][-1]:.1f}°  "
                      f"({len(m['f_clht']['x_alpha'])} points)")
                print(f"    HT  M : {m['f_clht']['y_mach'][0]:.3f} → "
                      f"{m['f_clht']['y_mach'][-1]:.3f}  "
                      f"({len(m['f_clht']['y_mach'])} points)")
                print()

            elif args.cmd in ("clwb", "cdwb", "cmwb", "cl_ht", "cd_ht", "cm_ht",
                              "clt", "cdt", "cmt", "all"):
                if _model[0] is None:
                    print("  Aucun modèle chargé — tapez 'load' d'abord.\n")
                    continue
                delta_it = getattr(args, 'delta_it', 0.0)
                fn_map = {
                    "clwb":  ("CL_wb",  lambda m, a, M: mod_aero.get_cl_wb(m, a, M)),
                    "cdwb":  ("CD_wb",  lambda m, a, M: mod_aero.get_cd_wb(m, a, M)),
                    "cmwb":  ("Cm_wb",  lambda m, a, M: mod_aero.get_cm_wb(m, a, M)),
                    "cl_ht": ("CL_ht",  lambda m, a, M: mod_aero.get_cl_ht(m, a, M, delta_it=delta_it)),
                    "cd_ht": ("CD_ht",  lambda m, a, M: mod_aero.get_cd_ht(m, a, M, delta_it=delta_it)),
                    "cm_ht": ("Cm_ht",  lambda m, a, M: mod_aero.get_cm_ht(m, a, M, delta_it=delta_it)),
                    "clt":   ("CL_t",   lambda m, a, M: mod_aero.get_cl_total(m, a, M, delta_it=delta_it)),
                    "cdt":   ("CD_t",   lambda m, a, M: mod_aero.get_cd_total(m, a, M, delta_it=delta_it)),
                    "cmt":   ("CM_t",   lambda m, a, M: mod_aero.get_cm_total(m, a, M, delta_it=delta_it)),
                }
                if args.cmd == "all":
                    _print_aero_all(_model[0], args.alpha, args.mach,
                                    delta_it=args.delta_it)
                else:
                    label, fn = fn_map[args.cmd]
                    _print_aero_result(label, fn(_model[0], args.alpha, args.mach),
                                       args.alpha, args.mach)

            elif args.cmd == "downwash":
                eps = mod_aero.f_downwash(args.alpha)
                print()
                print("=" * 50)
                print(f"  α = {args.alpha:.2f} °")
                print("─" * 50)
                print(f"  ε  = ε0 + εα·α")
                print(f"     = {mod_aero._EPS0} + {mod_aero._EPS_ALPHA}×{args.alpha:.2f}")
                print(f"  ε  =  {eps:>10.4f}  deg")
                print("=" * 50)
                print()

            elif args.cmd == "plot":
                if _model[0] is None:
                    print("  Aucun modèle chargé — tapez 'load' d'abord.\n")
                    continue
                print("  Affichage des graphiques (fermer la fenêtre pour continuer)...")
                mod_aero.plot_aero_model(_model[0])

            elif args.cmd == "plot_total":
                if _model[0] is None:
                    print("  Aucun modèle chargé — tapez 'load' d'abord.\n")
                    continue
                print("  Affichage des coefficients totaux (fermer la fenêtre pour continuer)...")
                mod_aero.plot_total(_model[0], delta_it=args.delta_it)

            elif args.cmd == "geom":
                fname = os.path.expanduser(args.file) if args.file else mod_aero.DEFAULT_FILE_GEOM
                print("  Affichage de la géométrie (fermer la fenêtre pour continuer)...")
                mod_aero.show_geometry(fname)

            elif args.cmd == "info":
                if _model[0] is None:
                    print("  Aucun modèle chargé — tapez 'load' d'abord.\n")
                    continue
                m = _model[0]
                print()
                print("=" * 46)
                print("  Modèle aérodynamique chargé")
                print("─" * 46)
                for key in ("f_clwb", "f_cdwb", "f_cmwb", "f_clht", "f_cdht"):
                    g = m[key]
                    print(f"  {key:<8}  α [{g['x_alpha'][0]:+.1f}°, {g['x_alpha'][-1]:+.1f}°]"
                          f"  M [{g['y_mach'][0]:.3f}, {g['y_mach'][-1]:.3f}]"
                          f"  ({g['value'].shape[0]}×{g['value'].shape[1]})")
                print("=" * 46)
                print()

        except (ValueError, FileNotFoundError) as e:
            print(f"  Erreur : {e}\n")
        except SystemExit:
            pass


# ============================== MODULE PROPULSION =============================

def print_prop_menu():
    print()
    print("─" * 62)
    print(f"  {mod_prop.NOM}")
    print("─" * 62)
    print("  thrust --n1 <N1> --mach <M> --h <alt> [--disa <v>]")
    print("         Poussée nette FN d'un moteur")
    print()
    print("  wf     --n1 <N1> --mach <M> --h <alt> [--disa <v>]")
    print("         Débit carburant WF d'un moteur")
    print()
    print("  ei     --n1 <N1> --mach <M> --h <alt> [--disa <v>]")
    print("         Indices d'émission OACI généralisés (méthode BFF)")
    print()
    print("  emis   --n1 <N1> --mach <M> --h <alt> [--disa <v>] [--t <durée s>]")
    print("         Masses de polluants émises par un moteur")
    print()
    print("  all    --n1 <N1> --mach <M> --h <alt> [--disa <v>]")
    print("         Poussée, débit carburant et indices d'émission")
    print()
    print("  plot   --h <alt> [--disa <v>]")
    print("         Graphiques 2D et 3D de FN et WF vs N1 et Mach")
    print()
    print("  plot_emis --h <alt> [--disa <v>]")
    print("         Courbes des indices d'émission vs N1 et Mach")
    print()
    print("  help  Afficher cette aide")
    print("  back  Retour au menu principal")
    print("─" * 62)
    print()


def _build_prop_parser():
    parser = _SilentParser(prog="")
    sub    = parser.add_subparsers(dest="cmd")
    sub.required = True

    for cmd in ("thrust", "wf", "ei", "emis", "all"):
        p = sub.add_parser(cmd)
        p.add_argument("--n1",   type=float, required=True, metavar="N1")
        p.add_argument("--mach", type=float, required=True, metavar="MACH")
        p.add_argument("--h",    type=float, required=True, metavar="ALTITUDE")
        p.add_argument("--disa", type=float, default=0.0,   metavar="ΔISA")
        if cmd == "emis":
            p.add_argument("--t", type=float, default=1.0,  metavar="DURÉE")

    for cmd in ("plot", "plot_emis"):
        p = sub.add_parser(cmd)
        p.add_argument("--h",    type=float, required=True, metavar="ALTITUDE")
        p.add_argument("--disa", type=float, default=0.0,   metavar="ΔISA")

    return parser


def loop_prop():
    """Boucle interactive du module de propulsion."""
    print_prop_menu()
    parser = _build_prop_parser()

    while True:
        try:
            ligne = input("  PROP> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not ligne:
            continue
        if ligne.lower() in ("back", "menu", "b"):
            break
        if ligne.lower() in ("help", "aide", "?"):
            print_prop_menu()
            continue

        try:
            args = parser.parse_args(shlex.split(ligne))

            if args.cmd == "plot":
                print("  Affichage des graphiques (fermer la fenêtre pour continuer)...")
                mod_prop.plot_prop(args.h, delta_isa=args.disa)
            elif args.cmd == "plot_emis":
                print("  Affichage des graphiques (fermer la fenêtre pour continuer)...")
                mod_prop.plot_emissions(args.h, delta_isa=args.disa)
            else:
                print()
                print("=" * 52)
                print(f"  N1 = {args.n1:.2f}    Mach = {args.mach:.4f}"
                      f"    h = {args.h:.0f} m    ΔISA = {args.disa:+.1f} °C")
                print("─" * 52)
                if args.cmd in ("thrust", "all"):
                    fn = mod_prop.get_thrust(args.n1, args.mach, args.h, args.disa)
                    print(f"  {'FN':<10} = {fn:>16.6f}")
                if args.cmd in ("wf", "all"):
                    wf = mod_prop.get_fuel_flow(args.n1, args.mach, args.h, args.disa)
                    print(f"  {'WF':<10} = {wf:>16.6f}")
                if args.cmd in ("ei", "all"):
                    ei = mod_prop.get_emission_indices(args.n1, args.mach,
                                                       args.h, args.disa)
                    print(f"  {'EI_NOx':<10} = {ei['EI_NOx']:>16.6f}  g/kg")
                    print(f"  {'EI_UHC':<10} = {ei['EI_UHC']:>16.6f}  g/kg")
                    print(f"  {'EI_CO':<10} = {ei['EI_CO']:>16.6f}  g/kg")
                    print(f"  {'EI_CO2':<10} = {ei['EI_CO2']:>16.6f}  kg/kg")
                if args.cmd == "emis":
                    em = mod_prop.get_emissions(args.n1, args.mach, args.h,
                                                args.disa, duration=args.t)
                    print(f"  Durée = {args.t:.1f} s    "
                          f"Carburant brûlé = {em['fuel_burn']:.4f} kg")
                    print("─" * 52)
                    print(f"  {'m_NOx':<10} = {em['m_NOx']:>16.6f}  g")
                    print(f"  {'m_UHC':<10} = {em['m_UHC']:>16.6f}  g")
                    print(f"  {'m_CO':<10} = {em['m_CO']:>16.6f}  g")
                    print(f"  {'m_CO2':<10} = {em['m_CO2']:>16.6f}  kg")
                print("=" * 52)
                print()

        except (ValueError, FileNotFoundError) as e:
            print(f"  Erreur : {e}\n")
        except SystemExit:
            pass


# ============================== MODULE ÉQUILIBRAGE (TRIM) =====================

def print_trim_menu():
    print()
    print("─" * 64)
    print(f"  {mod_trim.NOM}")
    print("─" * 64)
    print("  Équilibrage longitudinal en palier — résout (α, δstab, F_N) puis")
    print("  remonte au régime N1 et au débit carburant W_F.")
    print()
    print("  trim --mass <kg> --mach <M> --h <alt> [--disa <v>] [--xcg <f>] [--gamma <d>]")
    print("       [--eps-alpha <d>] [--eps-fn <n>] [--eps-dstab <d>]")
    print("       Équilibre l'avion pour la masse, le Mach et l'altitude donnés")
    print("       --xcg       : centrage (fraction de MAC, défaut 0.40)")
    print("       --gamma     : pente de trajectoire [deg] (défaut 0 = palier)")
    print("       --eps-alpha : tolérance convergence sur α      [deg] (défaut 1e-3)")
    print("       --eps-fn    : tolérance convergence sur F_N     [N]   (défaut 10)")
    print("       --eps-dstab : tolérance convergence sur δstab   [deg] (défaut 1e-3)")
    print()
    print("  Exemple :  trim --mass 500000 --mach 0.80 --h 10000")
    print()
    print("  info   Constantes du modèle (φ_T, positions moteurs, F_N^max, W_F^max…)")
    print("  help   Afficher cette aide")
    print("  back   Retour au menu principal")
    print("─" * 64)
    print()


def _build_trim_parser():
    parser = _SilentParser(prog="")
    sub    = parser.add_subparsers(dest="cmd")
    sub.required = True

    p = sub.add_parser("trim")
    p.add_argument("--mass",  type=float, required=True, metavar="MASSE")
    p.add_argument("--mach",  type=float, required=True, metavar="MACH")
    p.add_argument("--h",     type=float, required=True, metavar="ALTITUDE")
    p.add_argument("--disa",  type=float, default=0.0,   metavar="ΔISA")
    p.add_argument("--xcg",   type=float, default=0.40,  metavar="FRAC")
    p.add_argument("--gamma", type=float, default=0.0,   metavar="DEG")
    p.add_argument("--eps-alpha", type=float, default=mod_trim.EPS_ALPHA, metavar="DEG")
    p.add_argument("--eps-fn",    type=float, default=mod_trim.EPS_FN,    metavar="N")
    p.add_argument("--eps-dstab", type=float, default=mod_trim.EPS_DSTAB, metavar="DEG")

    sub.add_parser("info")
    return parser


def _print_trim_result(r, args):
    """Affiche le résultat d'un équilibrage."""
    statut = "convergé" if r['converged'] else "NON convergé"
    print()
    print("=" * 56)
    print(f"  Équilibrage  |  m = {args.mass/1000:.1f} t   M = {args.mach:.3f}   "
          f"h = {args.h:.0f} m")
    print(f"  ΔISA = {args.disa:+.1f} °C   x_cg = {args.xcg:.2f} MAC   "
          f"γ = {args.gamma:+.1f} °")
    print(f"  ε_α = {r['eps_alpha']:g} °   ε_FN = {r['eps_fn']:g} N   "
          f"ε_δstab = {r['eps_dstab']:g} °")
    print(f"  {statut} en {r['iterations']} itérations")
    print("─" * 56)
    print(f"  {'α (incidence)':<22} = {r['alpha']:>10.3f}  °")
    print(f"  {'δstab (calage THS)':<22} = {r['dstab']:>10.3f}  °")
    print("─" * 56)
    print(f"  {'CL':<22} = {r['CL']:>10.4f}")
    print(f"  {'CD':<22} = {r['CD']:>10.5f}")
    print(f"  {'Finesse L/D':<22} = {r['finesse']:>10.2f}")
    print(f"  {'Portance L':<22} = {r['L']:>10.0f}  N")
    print(f"  {'Traînée D':<22} = {r['D']:>10.0f}  N")
    print("─" * 56)
    print(f"  {'Poussée F_N (total)':<22} = {r['FN']:>10.0f}  N   "
          f"({r['FN']/1000:.1f} kN)")
    print(f"  {'F_N par moteur':<22} = {r['FN_engine']:>10.0f}  N   "
          f"({r['FN_engine']/1000:.1f} kN)")
    if r.get('thrust_limited'):
        print(f"  {'Régime N1':<22} = {'N/A':>10}     (LIMITÉ EN POUSSÉE :")
        print(f"  {'Débit W_F (total)':<22} = {'N/A':>10}      poussée requise > F_N max dispo)")
    else:
        print(f"  {'Régime N1':<22} = {r['N1']:>10.2f}  %")
        print(f"  {'Débit W_F (total)':<22} = {r['WF_total']:>10.4f}  kg/s "
              f"({r['WF_total_kgh']:.0f} kg/h)")
    print("=" * 56)
    print()
    _print_trim_history(r['history'], r['converged'])


# Codes ANSI pour surligner la ligne d'équilibre dans le tableau d'itérations
_GREEN = "\033[1;92m"   # vert vif (gras)
_RESET = "\033[0m"


def _print_trim_history(history, converged=True):
    """Affiche le détail de chaque itération de l'algorithme d'équilibrage.

    La dernière ligne (point d'équilibre) est surlignée en vert si l'algorithme
    a convergé.
    """
    print("  Détail des itérations (algorithme de Ghazi & Botez)")
    print("  " + "─" * 74)
    print(f"  {'it':>2}{'α[°]':>8}{'δstab[°]':>10}{'F_N[kN]':>10}"
          f"{'CL':>9}{'CD':>9}{'|Δα|':>9}{'|ΔF_N|':>9}{'|Δδ|':>9}")
    print("  " + "─" * 74)
    for i, h in enumerate(history):
        if h['it'] == 0:
            # Estimé initial : pas encore de CL/CD ni d'écarts
            ligne = (f"  {h['it']:>2}{h['alpha']:>8.3f}{h['dstab']:>10.3f}"
                     f"{h['FN']/1000:>10.1f}{'—':>9}{'—':>9}"
                     f"{'—':>9}{'—':>9}{'—':>9}")
        else:
            ligne = (f"  {h['it']:>2}{h['alpha']:>8.3f}{h['dstab']:>10.3f}"
                     f"{h['FN']/1000:>10.1f}{h['CL']:>9.4f}{h['CD']:>9.5f}"
                     f"{h['d_alpha']:>9.1e}{h['d_FN']:>9.1e}{h['d_dstab']:>9.1e}")
        # Ligne d'équilibre (dernière itération convergée) en vert
        if converged and i == len(history) - 1:
            ligne = _GREEN + ligne + _RESET
        print(ligne)
    print("  " + "─" * 74)
    if converged:
        print(f"  {_GREEN}■{_RESET} ligne verte = itération d'équilibre")
    print()


def loop_trim():
    """Boucle interactive du module d'équilibrage."""
    print_trim_menu()
    parser = _build_trim_parser()
    _model = [None]   # modèle aéro conservé entre les commandes

    while True:
        try:
            ligne = input("  TRIM> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not ligne:
            continue
        if ligne.lower() in ("back", "menu", "b"):
            break
        if ligne.lower() in ("help", "aide", "?"):
            print_trim_menu()
            continue

        try:
            args = parser.parse_args(shlex.split(ligne))

            if args.cmd == "info":
                print()
                print("=" * 56)
                print("  Constantes du modèle d'équilibrage")
                print("─" * 56)
                print(f"  φ_T (inclinaison poussée)   = {mod_trim.PHI_T:.2f} °")
                print(f"  c̄_w (corde aéro moyenne)    = {mod_trim.C_W:.2f} m")
                print(f"  Moteurs 2-3 (intérieurs)    x = {mod_trim.X_ENG_23:.2f} m   "
                      f"z = {mod_trim.Z_ENG_23:.2f} m")
                print(f"  Moteurs 1-4 (extérieurs)    x = {mod_trim.X_ENG_14:.2f} m   "
                      f"z = {mod_trim.Z_ENG_14:.2f} m")
                print(f"  F_N^max (par moteur)        = {mod_trim.FN_MAX/1000:.2f} kN")
                print(f"  W_F^max (par moteur)        = {mod_trim.WF_MAX_KGH:.1f} kg/h")
                print(f"  Convergence : ε_α = {mod_trim.EPS_ALPHA:g} °   "
                      f"ε_δstab = {mod_trim.EPS_DSTAB:g} °   ε_FN = {mod_trim.EPS_FN:g} N")
                print("=" * 56)
                print()
                continue

            # args.cmd == "trim"
            if _model[0] is None:
                print("  Chargement du modèle aérodynamique ...")
                _model[0] = mod_aero.build_aero_model()
            r = mod_trim.trim(args.mass, args.mach, args.h,
                              delta_isa=args.disa, x_cg=args.xcg,
                              gamma=args.gamma, model=_model[0],
                              eps_alpha=args.eps_alpha, eps_fn=args.eps_fn,
                              eps_dstab=args.eps_dstab)
            _print_trim_result(r, args)

        except (ValueError, FileNotFoundError) as e:
            print(f"  Erreur : {e}\n")
        except SystemExit:
            pass


# ============================== MODULE PERFORMANCE (PERF) =====================

def print_perf_menu():
    print()
    print("─" * 64)
    print(f"  {mod_perf.NOM}")
    print("─" * 64)
    print("  Vitesses de croisière optimales à masse et altitude fixées :")
    print("    MRC  — portée spécifique maximale (SR = TAS / W_F)")
    print("    LRC  — vitesse rapide à SR = 0.99·SR_max (convention industrielle)")
    print("    ECON — coût minimal carburant + temps (via Cost Index)")
    print()
    print("  cruise --mass <kg> --h <alt> [--disa <v>] [--ci <kg/min>]")
    print("         [--mmin <M>] [--mmax <M>] [--n <pts>]")
    print("         Balaye le Mach (trim à chaque pas) et extrait MRC, LRC, ECON")
    print("         --ci    : Cost Index pour ECON [kg/min] (défaut 0 → ECON ≡ MRC)")
    print("         --mmin  : Mach min du balayage (défaut 0.50)")
    print("         --mmax  : Mach max du balayage (défaut 0.90)")
    print("         --n     : nombre de points d'échantillonnage (défaut 41)")
    print()
    print("  Exemple :  cruise --mass 450000 --h 10668 --ci 30")
    print()
    print("  help   Afficher cette aide")
    print("  back   Retour au menu principal")
    print("─" * 64)
    print()


def _build_perf_parser():
    parser = _SilentParser(prog="")
    sub    = parser.add_subparsers(dest="cmd")
    sub.required = True

    p = sub.add_parser("cruise")
    p.add_argument("--mass", type=float, required=True, metavar="MASSE")
    p.add_argument("--h",    type=float, required=True, metavar="ALTITUDE")
    p.add_argument("--disa", type=float, default=0.0,   metavar="ΔISA")
    p.add_argument("--ci",   type=float, default=0.0,   metavar="KG/MIN")
    p.add_argument("--mmin", type=float, default=mod_perf.MACH_MIN_DEFAUT, metavar="MACH")
    p.add_argument("--mmax", type=float, default=mod_perf.MACH_MAX_DEFAUT, metavar="MACH")
    p.add_argument("--n",    type=int,   default=mod_perf.N_PTS_DEFAUT,    metavar="PTS")
    return parser


def _print_perf_result(r, args):
    """Affiche les vitesses de croisière optimales MRC / LRC / ECON."""
    print()
    print("=" * 64)
    print(f"  Vitesses de croisière optimales  |  m = {args.mass/1000:.1f} t   "
          f"h = {args.h:.0f} m")
    print(f"  ΔISA = {args.disa:+.1f} °C   Cost Index = {args.ci:.0f} kg/min   "
          f"balayage M{args.mmin:.2f}→{args.mmax:.2f} ({args.n} pts)")
    print("─" * 64)
    if r['MRC'] is None:
        print("  Aucun point exploitable : l'avion est limité en poussée sur tout")
        print("  l'intervalle de Mach demandé (W_F non défini).")
        print("=" * 64)
        print()
        return
    print(f"  Portée spécifique maximale (SR_max) = {r['sr_max']:.1f} m/kg "
          f"({r['sr_max']/1852.0:.4f} NM/kg)")
    print("─" * 64)
    print(f"  {'':<6}{'Mach':>8}{'TAS[kt]':>10}{'SR[m/kg]':>11}"
          f"{'SR[NM/kg]':>11}{'W_F[kg/s]':>11}{'L/D':>8}")
    print("  " + "─" * 62)
    for cle in ("MRC", "LRC", "ECON"):
        o = r[cle]
        if o is None:
            print(f"  {cle:<6}{'—':>8}")
            continue
        print(f"  {cle:<6}{o['mach']:>8.4f}{o['tas_kt']:>10.1f}{o['sr']:>11.1f}"
              f"{o['sr_nm_per_kg']:>11.4f}{o['wf']:>11.4f}{o['finesse']:>8.2f}")
    print("=" * 64)
    if args.ci == 0:
        print("  Note : Cost Index = 0 → ECON coïncide avec MRC (coût = carburant seul).")
    print()


def loop_perf():
    """Boucle interactive du module de performance de croisière."""
    print_perf_menu()
    parser = _build_perf_parser()
    _model = [None]   # modèle aéro conservé entre les commandes

    while True:
        try:
            ligne = input("  PERF> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not ligne:
            continue
        if ligne.lower() in ("back", "menu", "b"):
            break
        if ligne.lower() in ("help", "aide", "?"):
            print_perf_menu()
            continue

        try:
            args = parser.parse_args(shlex.split(ligne))

            # args.cmd == "cruise"
            if _model[0] is None:
                print("  Chargement du modèle aérodynamique ...")
                _model[0] = mod_aero.build_aero_model()
            print("  Balayage du Mach en cours ...")
            r = mod_perf.cruise_speeds(
                args.mass, args.h, delta_isa=args.disa, cost_index=args.ci,
                mach_min=args.mmin, mach_max=args.mmax, n_pts=args.n,
                model=_model[0])
            _print_perf_result(r, args)

        except (ValueError, FileNotFoundError) as e:
            print(f"  Erreur : {e}\n")
        except SystemExit:
            pass


# ============================== MODULES EN DÉVELOPPEMENT ======================

def loop_dev(module):
    """Affiche un message pour les modules non encore implémentés."""
    print()
    print(f"  {module.NOM}")
    print(f"  Ce module est en cours de développement.")
    print()


# ---------------------------------------------------------------------------
# Table de dispatch : associe chaque clé à sa boucle interactive
# ---------------------------------------------------------------------------

_LOOPS = {
    "atm":  loop_atm,
    "conv": loop_conv,
    "aero": loop_aero,
    "prop": loop_prop,
    "trim": loop_trim,
    "perf": loop_perf,
}


def main_loop():
    """Boucle principale du menu — gère la navigation entre modules."""
    print_main_menu()

    while True:
        try:
            choix = input("  MENU> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Au revoir.")
            break

        if not choix:
            continue
        if choix in ("quit", "exit", "q"):
            print("  Au revoir.")
            break
        if choix in ("help", "?"):
            print_main_menu()
            continue

        # Résolution de la clé par numéro (1–5) ou par nom abrégé (atm, conv…)
        cle = None
        for m in MODULES:
            if choix == str(m["id"]) or choix == m["cle"]:
                cle = m["cle"]
                break

        if cle is None:
            print(f"  Module '{choix}' inconnu. Tapez un numéro (1–5) ou une clé.\n")
            continue

        # Lancement de la boucle du module sélectionné
        _LOOPS[cle]()
        # Réaffichage du menu principal au retour du module
        print_main_menu()


# ---------------------------------------------------------------------------
# Mode direct (arguments passés directement au lancement)
# ---------------------------------------------------------------------------

def main():
    # Si aucun argument n'est passé, on entre en mode interactif
    if len(sys.argv) == 1:
        main_loop()
        return

    # Mode direct : python main.py <cle_module> <commande> [args...]
    cle   = sys.argv[1].lower()
    reste = sys.argv[2:]  # arguments transmis au parser du module

    if cle == "atm":
        parser = _build_atm_parser()
        try:
            args = parser.parse_args(reste)
            if args.h is not None:
                # Validation de la plage d'altitude
                if not (0 <= args.h <= 20000):
                    raise ValueError("L'altitude --h doit être comprise entre 0 et 20 000 m.")
                h = args.h
            else:
                # Inversion pression → altitude
                h = mod_atm.altitude_from_pressure(args.p, args.disa)
                if h > 20000:
                    raise ValueError(f"La pression correspond à h = {h:.0f} m > 20 000 m.")
            print_atm_single(h, args.disa)
        except ValueError as e:
            print(f"Erreur : {e}", file=sys.stderr)
            sys.exit(1)

    elif cle == "conv":
        parser = _build_conv_parser()
        try:
            args = parser.parse_args(reste)
            if args.cmd == "mach2tas":
                tas = mod_conv.mach_to_tas(args.m, args.h, args.disa)
                _print_conv_aero("Mach", args.m, "TAS", tas, args.m, args.h, args.disa)
            elif args.cmd == "tas2mach":
                tas_ms = _to_ms(args.v, args.unite)
                M = mod_conv.tas_to_mach(tas_ms, args.h, args.disa)
                _print_conv_aero("TAS", tas_ms, "Mach", M, M, args.h, args.disa)
            elif args.cmd == "mach2cas":
                cas = mod_conv.mach_to_cas(args.m, args.h, args.disa)
                _print_conv_aero("Mach", args.m, "CAS", cas, args.m, args.h, args.disa)
            elif args.cmd == "cas2mach":
                cas_ms = _to_ms(args.v, args.unite)
                M = mod_conv.cas_to_mach(cas_ms, args.h, args.disa)
                _print_conv_aero("CAS", cas_ms, "Mach", M, M, args.h, args.disa)
            elif args.cmd == "tas2cas":
                tas_ms = _to_ms(args.v, args.unite)
                cas_ms = mod_conv.tas_to_cas(tas_ms, args.h, args.disa)
                M = mod_conv.tas_to_mach(tas_ms, args.h, args.disa)
                _print_conv_aero("TAS", tas_ms, "CAS", cas_ms, M, args.h, args.disa)
            elif args.cmd == "cas2tas":
                cas_ms = _to_ms(args.v, args.unite)
                tas_ms = mod_conv.cas_to_tas(cas_ms, args.h, args.disa)
                M = mod_conv.cas_to_mach(cas_ms, args.h, args.disa)
                _print_conv_aero("CAS", cas_ms, "TAS", tas_ms, M, args.h, args.disa)
        except ValueError as e:
            print(f"Erreur : {e}", file=sys.stderr)
            sys.exit(1)

    else:
        print(f"Module '{cle}' inconnu ou non disponible en mode direct.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
