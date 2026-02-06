import time

def micros():
    return round(time.time() * 1e6)

def add_padding_zeros(c, n):
    # ajout de zéros devant le string c jusqu'à atteindre une longueur >= n
    while len(c) < n:
        c = '0' + c
    return c