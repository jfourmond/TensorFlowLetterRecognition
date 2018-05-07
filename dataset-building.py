"""
    Data building for real image recognition
"""
import os
import random as rd
import shutil as sl
import time

LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
           'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'empty']

INPUT_DIRECTORY = "data/letter-recognition/"

OUTPUT_TRAIN_DIRECTORY = "data/train/"
OUTPUT_TEST_DIRECTORY = "data/test/"
OUTPUT_PREDICT_DIRECTORY = "data/predict/"

TRAIN_PRCT = 0.7 # 70% de données d'entraînement, 30% de données de test

def fill_directory(directory, data):
    if os.path.exists(directory):
        sl.rmtree(directory)
    os.makedirs(directory)
    for letter in LETTERS:
        os.makedirs(directory + letter)
    # Remplissage
    for d in data:
        input_file = d[0] + d[1] + "/" + d[2]
        output_file = directory+ d[1] + "/" + d[2]
        sl.copy(input_file, output_file)

if __name__ == "__main__":
    start = time.time()

    polices = os.listdir(INPUT_DIRECTORY)

    files = []
    for police in polices:
        aux = os.listdir(INPUT_DIRECTORY + police)
        files.extend([(INPUT_DIRECTORY, police, a) for a in aux])

    rd.shuffle(files)

    n_train = (int)(len(files) * TRAIN_PRCT)

    train_data = files[0:n_train]
    test_data = files[n_train:len(files)]

    # Création du répertoire "data/train/"
    fill_directory(OUTPUT_TRAIN_DIRECTORY, train_data)
    # Création du répertoire "data/test/"
    fill_directory(OUTPUT_TEST_DIRECTORY, test_data)
    # Création du répertoire "data/predict"
    fill_directory(OUTPUT_PREDICT_DIRECTORY, rd.sample(test_data, 10))

    end = time.time() - start
    minutes, seconds = divmod(end, 60)
    print("Temps d'exécution : {}m {:.4f}s".format(int(minutes), seconds))

