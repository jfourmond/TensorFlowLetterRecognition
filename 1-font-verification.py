"""
    font-verification.py : 
        Affiche les différentes polices, exclues les polices dont le chargement échoue
        et laisse le choix à l'utilisateur d'exclure certaines polices.
        Les polices exclues seront stockées dans le fichier 'data/excluded-fonts.json'
    Commandes :
        - 'q' et 'ESC' pour quitter l'interface
        - 'e' pour ajouter la police courante à liste des polices à exclure
        - Flèches HAUT et BAS pour changer de police
        - Flèches GAUCHE et DROITE pour changer de lettre
"""
import argparse
import cv2
import json
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import string
import time

FLAGS = None
SIZE = (20, 20)
LETTER_POS = (3, 0)

ARROWS = { 2490368:'up', 2555904:'right', 2621440:'down', 2424832:'left' }

FONT_PATH = "C:\\Windows\\Fonts" # Windows Font Location
EXCLUDED_FONTS_JSON = "data/excluded-fonts.json"

def main():
    start = time.time()

    # Création ou suppression du répertoire d'exportation
    if not os.path.exists(EXCLUDED_FONTS_JSON):
        with open(EXCLUDED_FONTS_JSON, 'w') as outfile:
            json.dump([ ], outfile)
    
    excluded_fonts = json.loads(open(EXCLUDED_FONTS_JSON).read())

    # Listing des polices du système
    font_files = os.listdir(FONT_PATH)
    # Filtrage sur les polices ttf (not fon)
    font_files = [font_file for font_file in font_files if not font_file.lower().endswith('.fon')]
    # Filtrage des polices exclues
    font_files = [font_file for font_file in font_files if not font_file in excluded_fonts]
    # Comptage des fichiers de police
    count = len(font_files)
    # Listing des lettres capitales
    letters = string.ascii_uppercase

    font_i = 0
    letter_i = 0
    font_file = ''
    font_name = ''

    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    while(1):
        key = cv2.waitKeyEx()

        if key == 113 or key == 27:
            break
        if key == 101:
            # Exclusion
            print('Added in excluded police : {}'.format(font_name))
            font_files.remove(font_file)
            if not font_file in excluded_fonts:
                excluded_fonts.append(font_file)

        arrow = ARROWS.get(key)
        if arrow == 'up':
            font_i = font_i + 1
            if font_i >= len(font_files): font_i = 0
        elif arrow == 'down' :
            font_i = font_i - 1
            if font_i < 0 : font_i = len(font_files) - 1
        elif arrow == 'left' :
            letter_i = letter_i - 1
            if letter_i < 0 : letter_i = len(letters) - 1
        elif arrow == 'right' :
            letter_i = letter_i + 1
            if letter_i >= len(letters): letter_i = 0

        # Choix de la police
        font_file = font_files[font_i]
        font_path = "{}/{}".format(FONT_PATH, font_file)
        n = font_file.find('.')
        font_name = font_file[0:n]
        # Choix de la lettre
        letter = letters[letter_i]

        # Chargement de la police
        try:
            font = ImageFont.truetype(font_path, 16)
            print("FONT : {}".format(font_name))
        except OSError:
            print("Failure while loading '{}', added in excluded police.".format(font_file))
            font_files.remove(font_file)
            if not font_file in excluded_fonts:
                excluded_fonts.append(font_file)

        im = Image.new('RGB', SIZE, (255,255,255))
        draw = ImageDraw.Draw(im)
        draw.text(LETTER_POS, letter, font=font, fill="black")

        # Affichage de l'image & de la section
        cv2.imshow('image', np.array(im))

    cv2.destroyAllWindows()

    print("Font Files : {}".format(count))
    print("Excluded Fonts : {}".format(len(excluded_fonts)))

    with open(EXCLUDED_FONTS_JSON, 'w') as outfile:
            json.dump(excluded_fonts, outfile)

    end = time.time() - start
    minutes, seconds = divmod(end, 60)
    print("Execution Time : {}m {:.4f}s".format(int(minutes), seconds))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    FLAGS, unparsed = parser.parse_known_args()
    main()