"""Versions PNG à fond transparent des deux figures sans rendu vectoriel.

`fig_geometry` (maillage OpenVSP) et `fig_aero_surfaces` (surfaces 3D) ne sont
produites qu'en PDF par make_figures.py — matplotlib ne sort pas de SVG
exploitable pour ces rendus 3D. La page « Présentation » de l'application
(app.py) a besoin d'un format que le navigateur affiche : on les convertit une
fois pour toutes en PNG, avec un fond transparent pour qu'elles se posent sur le
verre dépoli des cartes comme les autres figures.

Le PDF contient un rectangle blanc peint (facecolor par défaut de matplotlib) :
l'option -transp de pdftocairo ne suffit donc pas, on détoure ensuite le fond
par remplissage depuis les quatre coins. Le blanc INTÉRIEUR aux tracés (panneaux
des axes 3D) est préservé, puisqu'il n'est pas connecté au bord.

Prérequis : poppler (pdftocairo) et Pillow.
Usage : python rapport/figures/make_png.py
"""

import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

FIGDIR = Path(__file__).parent
SORTIE = FIGDIR / "png"
FIGURES = ("fig_geometry", "fig_aero_surfaces")
DPI = 170            # ~1200 px de large : net sur un vidéoprojecteur
SEUIL = 20           # tolérance du remplissage (bords antialiasés)


def convertir(nom):
    src = FIGDIR / f"{nom}.pdf"
    if not src.exists():
        raise SystemExit(f"{src} est absent — lancer make_figures.py d'abord.")

    with tempfile.TemporaryDirectory() as tmp:
        prefixe = Path(tmp) / nom
        subprocess.run(["pdftocairo", "-png", "-transp", "-r", str(DPI),
                        "-singlefile", str(src), str(prefixe)], check=True)
        im = Image.open(f"{prefixe}.png").convert("RGBA")

    for coin in ((0, 0), (im.width - 1, 0), (0, im.height - 1),
                 (im.width - 1, im.height - 1)):
        ImageDraw.floodfill(im, coin, (255, 255, 255, 0), thresh=SEUIL)

    SORTIE.mkdir(exist_ok=True)
    im.save(SORTIE / f"{nom}.png")
    opaque = sum(im.getchannel("A").histogram()[1:]) / (im.width * im.height)
    print(f"{nom}.png — {im.width}×{im.height}, {opaque:.0%} de pixels opaques")


if __name__ == "__main__":
    for figure in FIGURES:
        convertir(figure)
