"""
    data-creation.py : 
        Créer une multitude d'images à partir des polices disponibles sur la
        machine (dans le répertoire "C:\\Windows\\Fonts"), qui seront stockées
        dans le répertoire : data/letter-recognition/.

        Les polices dans le fichier json 'data/excluded-fonts.json' seront
        exclues du traitement.
"""
import argparse
import json
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import shutil as sl
import string
import time

FLAGS = None
SIZE = (20, 20)
LETTER_POS = (3, 0)

EXCLUDED_FONTS_JSON = "data/excluded-fonts.json"

FONT_PATH = "C:\\Windows\\Fonts" # Windows Font Location
EXPORT_DIR = "data/letter-recognition"

def main():
    start = time.time()

    excluded_fonts = [ ]

    if os.path.exists(EXCLUDED_FONTS_JSON):
        excluded_fonts = json.loads(open(EXCLUDED_FONTS_JSON).read())

    # Création ou suppression du répertoire d'exportation
    if os.path.exists(EXPORT_DIR):
        sl.rmtree(EXPORT_DIR)
    os.makedirs(EXPORT_DIR)
    # Création des répertoires d'exportation des lettres
    [os.mkdir("{}/{}".format(EXPORT_DIR, letter)) for letter in string.ascii_lowercase]
    # + Répertoire du caractère vide
    os.mkdir("{}/{}".format(EXPORT_DIR, 'empty'))

    # Compte des erreurs et des succès
    error = 0
    success = 0
    printed = 0
    # Listing des polices du système
    font_files = os.listdir(FONT_PATH)
    # Filtrage sur les polices ttf (not fon)
    font_files = [font_file for font_file in font_files if not font_file.lower().endswith('.fon')]
    # Filtrage des polices exclues
    font_files = [font_file for font_file in font_files if not font_file in excluded_fonts]
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
            # Création des lettres minuscules
            for letter in string.ascii_lowercase:
                export_path = "{}/{}/{}_lower.png".format(EXPORT_DIR, letter, font_name)
                im = Image.new('RGB', SIZE, (255,255,255))
                draw = ImageDraw.Draw(im)
                draw.text(LETTER_POS, letter, font=font, fill="black")
                im.save(export_path, "PNG")
                printed = printed + 1
            # Création des lettres majuscules
            for letter in string.ascii_uppercase:
                export_path = "{}/{}/{}_upper.png".format(EXPORT_DIR, letter, font_name)
                im = Image.new('RGB', SIZE, (255,255,255))
                draw = ImageDraw.Draw(im)
                draw.text(LETTER_POS, letter, font=font, fill="black")
                im.save(export_path, "PNG")
                printed = printed + 1
            # Ajout du caractère vide
            im = Image.new('RGB', SIZE, (255,255,255))
            export_path = "{}/{}/{}.png".format(EXPORT_DIR, 'empty', font_name)
            im.save(export_path, "PNG")
            printed = printed + 1
            
        except OSError as ose:
            print("Error : {}".format(ose))
            error = error + 1

    print("Loading Error : {}".format(error))
    print("Loading Success : {}".format(success))
    print("Font Files : {}".format(count))
    print("Saved Images : {}".format(printed))

    end = time.time() - start
    minutes, seconds = divmod(end, 60)
    print("Execution Time : {}m {:.4f}s".format(int(minutes), seconds))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    FLAGS, unparsed = parser.parse_known_args()
    main()