"""
    data-creation.py : 
        Créer une multitude d'images à partir des polices disponibles sur la
        machine (dans le répertoire "C:\\Windows\\Fonts"), qui seront stockées
        dans le répertoire : data/letter-recognition/
"""
import argparse
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import shutil as sl
import string
import time

FLAGS = None
SIZE = (20, 20)
LETTER_POS = (3, 0)

EXCLUDED_FONTS = [
    "BSSYM7.TTF", "flat_officeFontsPreview.ttf", "holomdl2.ttf", "marlett.ttf", "MTEXTRA.TTF", 
    "OFFSYM.TTF", "OFFSYMB.TTF", "OFFSYML.TTF", "OFFSYMSB.TTF", "OFFSYMSL.TTF", "OFFSYMXL.TTF",
    "OUTLOOK.TTF", "REFSPCL.TTF" "segmdl2.ttf","webdings.ttf", "wingding.ttf", "WINGDNG2.TTF",
    "WINGDNG3.TTF"]

FONT_PATH = "C:\\Windows\\Fonts" # Windows Font Location
EXPORT_DIR = "data/letter-recognition"

def main():
    start = time.time()

    # Création ou suppression du répertoire d'exportation
    if os.path.exists(EXPORT_DIR):
        sl.rmtree(EXPORT_DIR)
    os.makedirs(EXPORT_DIR)
    # Création des répertoires d'exportation des lettres
    [os.mkdir("{}/{}".format(EXPORT_DIR, letter)) for letter in string.ascii_lowercase]

    # Compte des erreurs et des succès
    error = 0
    success = 0
    # Listing des polices du système
    font_files = os.listdir(FONT_PATH)
    # Filtrage sur les polices ttf (not fon)
    font_files = [font_file for font_file in font_files if not font_file.lower().endswith('.fon')]
    # Filtrage des polices exclues
    font_files = [font_file for font_file in font_files if not font_file in EXCLUDED_FONTS]
    # Comptage des fichiers de police
    count = len(font_files)

    for font_file in font_files:
        font_path = "{}/{}".format(FONT_PATH, font_file)
        n = font_file.find('.')
        font_name = font_file[0:n]
        try:
            # Chargement de la police
            font = ImageFont.truetype(font_path, 16)
            print("FONT : {}".format(font_name))
            success = success + 1

            for letter in string.ascii_lowercase:
                export_path = "{}/{}/{}_lower.png".format(EXPORT_DIR, letter, font_name)
                im = Image.new('RGB', SIZE, (255,255,255))
                draw = ImageDraw.Draw(im)
                draw.text(LETTER_POS, letter, font=font, fill="black")
                im.save(export_path, "PNG")

            for letter in string.ascii_uppercase:
                export_path = "{}/{}/{}_upper.png".format(EXPORT_DIR, letter, font_name)
                im = Image.new('RGB', SIZE, (255,255,255))
                draw = ImageDraw.Draw(im)
                draw.text(LETTER_POS, letter, font=font, fill="black")
                im.save(export_path, "PNG")
        except OSError as ose:
            print("Error : {}".format(ose))
            error = error + 1

    print("Erreurs de chargement : {}".format(error))
    print("Succès de chargement : {}".format(success))
    print("Fichier de police : {}".format(count))

    end = time.time() - start
    minutes, seconds = divmod(end, 60)
    print("Temps d'exécution : {}m {:.4f}s".format(int(minutes), seconds))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    FLAGS, unparsed = parser.parse_known_args()
    main()