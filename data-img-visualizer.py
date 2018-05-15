import argparse
import cv2
import numpy as np
import os
from PIL import Image

FLAGS = None

DATA_DIR = "data/letter-recognition/"

def main():
    LETTER = FLAGS.letter

    # Récupération des images dans le répertoire
    files = os.listdir(DATA_DIR + LETTER)
    files = [DATA_DIR + LETTER + '/' + a for a in files]
    # Lecture des images
    images = [cv2.imread(file) for file in files]
    images = [cv2.copyMakeBorder(image, 1, 0, 1, 0, cv2.BORDER_CONSTANT) for image in images]

    images_splitted = np.array_split(np.array(images), len(files) / 30)
    max_width = images_splitted[0].shape[1] * images_splitted[0].shape[0]

    horizons = [ ]
    # Concaténation horizontale
    for i in range(0, len(images_splitted)):
        horizon = np.hstack(images_splitted[i])
        while(not horizon.shape[1] == max_width):
            # Ajout d'une image vide
            horizon = np.hstack([horizon, cv2.copyMakeBorder(np.array(Image.new('RGB', (20, 20), (255,255,255))), 1, 0, 1, 0, cv2.BORDER_CONSTANT)])
        horizons.append(horizon)
    # Concaténation verticale
    image = np.vstack(horizons)

    cv2.imshow('LETTER {}'.format(LETTER), image)

    cv2.waitKey()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'letter',
        type=str,
        help='Path to the model'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()