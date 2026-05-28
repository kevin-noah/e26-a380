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

# ---------------------------------------------------------------------------
# Registre des modules — chaque entrée porte son module et son statut
# ---------------------------------------------------------------------------
# registre des modules
MODULES = [
    {"id": 1, "cle": "atm",  "module": mod_atm,  "dispo": True},
    {"id": 2, "cle": "conv", "module": mod_conv,  "dispo": True},
    {"id": 3, "cle": "aero", "module": mod_aero,  "dispo": True},
    {"id": 4, "cle": "prop", "module": mod_prop,  "dispo": False},
    {"id": 5, "cle": "trim", "module": mod_trim,  "dispo": False},
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
        statut = "" if m["dispo"] else "  [en développement]"
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
    print("  clwb  --alpha <deg> --mach <M>   CL aile + fuselage")
    print("  cdwb  --alpha <deg> --mach <M>   CD aile + fuselage")
    print("  cmwb  --alpha <deg> --mach <M>   Cm aile + fuselage")
    print("  cl_ht --alpha <deg> --mach <M>   CL empennage")
    print("  cd_ht --alpha <deg> --mach <M>   CD empennage")
    print("  all   --alpha <deg> --mach <M>   Tous les coefficients")
    print()
    print("  downwash --alpha <deg>           Angle de downwash ε [deg]")
    print()
    print("  plot              Graphiques 2D et 3D des coefficients")
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

    for cmd in ("clwb", "cdwb", "cmwb", "cl_ht", "cd_ht", "all"):
        p = sub.add_parser(cmd)
        p.add_argument("--alpha", type=float, required=True)
        p.add_argument("--mach",  type=float, required=True)

    p = sub.add_parser("downwash")
    p.add_argument("--alpha", type=float, required=True)

    sub.add_parser("plot")
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


def _print_aero_all(model, alpha, mach):
    cl_wb = mod_aero.get_cl_wb(model, alpha, mach)
    cd_wb = mod_aero.get_cd_wb(model, alpha, mach)
    cm_wb = mod_aero.get_cm_wb(model, alpha, mach)
    cl_ht = mod_aero.get_cl_ht(model, alpha, mach)
    cd_ht = mod_aero.get_cd_ht(model, alpha, mach)
    print()
    print("=" * 46)
    print(f"  α = {alpha:.2f} °    Mach = {mach:.4f}")
    print("─" * 46)
    print(f"  {'CL_wb':<10} = {cl_wb:>14.8f}")
    print(f"  {'CD_wb':<10} = {cd_wb:>14.8f}")
    print(f"  {'Cm_wb':<10} = {cm_wb:>14.8f}")
    print(f"  {'CL_ht':<10} = {cl_ht:>14.8f}")
    print(f"  {'CD_ht':<10} = {cd_ht:>14.8f}")
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
                print(f"    α : {m['f_clwb']['x_alpha'][0]:.1f}° → "
                      f"{m['f_clwb']['x_alpha'][-1]:.1f}°  "
                      f"({len(m['f_clwb']['x_alpha'])} points)")
                print(f"    M : {m['f_clwb']['y_mach'][0]:.3f} → "
                      f"{m['f_clwb']['y_mach'][-1]:.3f}  "
                      f"({len(m['f_clwb']['y_mach'])} points)")
                print()

            elif args.cmd in ("clwb", "cdwb", "cmwb", "cl_ht", "cd_ht", "all"):
                if _model[0] is None:
                    print("  Aucun modèle chargé — tapez 'load' d'abord.\n")
                    continue
                fn_map = {
                    "clwb":  ("CL_wb",  mod_aero.get_cl_wb),
                    "cdwb":  ("CD_wb",  mod_aero.get_cd_wb),
                    "cmwb":  ("Cm_wb",  mod_aero.get_cm_wb),
                    "cl_ht": ("CL_ht",  mod_aero.get_cl_ht),
                    "cd_ht": ("CD_ht",  mod_aero.get_cd_ht),
                }
                if args.cmd == "all":
                    _print_aero_all(_model[0], args.alpha, args.mach)
                else:
                    label, fn = fn_map[args.cmd]
                    _print_aero_result(label, fn(_model[0], args.alpha, args.mach),
                                       args.alpha, args.mach)

            elif args.cmd == "downwash":
                eps = mod_aero.f_downwash(args.alpha)
                print()
                print("=" * 46)
                print(f"  α = {args.alpha:.2f} °")
                print("─" * 46)
                print(f"  ε  = ε0 + εα·α = {mod_aero._EPS0} + {mod_aero._EPS_ALPHA}×{args.alpha:.2f}")
                print(f"  ε  =  {eps:>10.4f}  deg")
                print("=" * 46)
                print()

            elif args.cmd == "plot":
                if _model[0] is None:
                    print("  Aucun modèle chargé — tapez 'load' d'abord.\n")
                    continue
                print("  Affichage des graphiques (fermer la fenêtre pour continuer)...")
                mod_aero.plot_aero_model(_model[0])

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
    "prop": lambda: loop_dev(mod_prop),
    "trim": lambda: loop_dev(mod_trim),
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
