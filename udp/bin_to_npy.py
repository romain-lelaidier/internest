import numpy as np
from read_bin_file import date_data_tab, to_human_readable

"""
Objectif :
Encoder les .bin dans un fichier lisible par Python pour le traitement du signal
"""


"""Fonctions reprises du script de Colin"""

def date_data_tab(file_path, verbose=False):
    with open(file_path, "rb") as fb:
        file = fb.read()
        dates = [
            file[CHUNK_SIZE // 8 * i : CHUNK_SIZE // 8 * i + 8]
            for i in range(CHUNK_PER_FILE)
        ]
        data = [
            file[CHUNK_SIZE // 8 * i + 8 : CHUNK_SIZE // 8 * (i + 1)]
            for i in range(CHUNK_PER_FILE)
        ]
    data = [
        [
            data[i][j : j + 1] for j in range(len(data[i]))
        ]  # data[i][j] converts the signed binary to int as if it were unsigned
        for i in range(len(data))
    ]
    if verbose:
        verbose_data_bin_tab(file, dates, data)

    return dates, np.array(data)


""" Pipeline """

# 1. Lire le binaire brut
bin_dates, bin_data = date_data_tab("enregistrement.bin")

# 2. Convertir en valeurs lisibles (votre fonction existante)
# Cela sépare clairement le temps et l'amplitude
dates_reelles, signal_audio = to_human_readable(bin_dates, bin_data)

# 3. Sauvegarder en format NumPy compressé (.npz)
# On sauve deux tableaux nommés : 'timestamps' et 'signal'
np.savez("signal_traite.npz", timestamps=np.array(dates_reelles), signal=signal_audio)
