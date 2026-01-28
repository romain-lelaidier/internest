import numpy as np
from read_bin_file import date_data_tab, to_human_readable

"""
Objectif :
Encoder les .bin dans un fichier lisible par Python pour le traitement du signal
"""

# 1. Lire le binaire brut
bin_dates, bin_data = date_data_tab("enregistrement.bin")

# 2. Convertir en valeurs lisibles (votre fonction existante)
# Cela sépare clairement le temps et l'amplitude
dates_reelles, signal_audio = to_human_readable(bin_dates, bin_data)

# 3. Sauvegarder en format NumPy compressé (.npz)
# On sauve deux tableaux nommés : 'timestamps' et 'signal'
np.savez("signal_traite.npz", timestamps=np.array(dates_reelles), signal=signal_audio)
