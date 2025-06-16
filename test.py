import numpy as np
import pandas as pd
from scipy.signal import convolve
import matplotlib.pyplot as plt

def wczytaj_sygnal(nazwa_pliku):
    try:
        dane = pd.read_csv(nazwa_pliku, header=None)
        return dane.squeeze().values  # konwertuj do 1D numpy array
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku '{nazwa_pliku}': {e}")
        exit()

def main():
    plik1 ='syg1_1_1024.csv'
    plik2 ='syg3_1_256.csv'

    sygnal1 = wczytaj_sygnal(plik1)
    sygnal2 = wczytaj_sygnal(plik2)

    wynik = convolve(sygnal1, sygnal2, mode='full')

    # Zapisz wynik do pliku
    wynik_df = pd.DataFrame(wynik, columns=['Amplituda'])
    wynik_df.to_csv("splot_wynik.csv", index=False)
    print("Wynik splotu zapisany do pliku 'splot_wynik.csv'.")

    # (Opcjonalnie) wykres
    plt.figure(figsize=(10, 4))
    plt.plot(wynik, label='Splot')
    plt.title("Wynik splotu sygnałów")
    plt.xlabel("Próbka")
    plt.ylabel("Amplituda")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
