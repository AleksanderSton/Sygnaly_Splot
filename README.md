# Analizator Sygnałów - Ręczny Splot

Program do analizy i obróbki sygnałów zapisanych w plikach CSV.  
Pozwala na wczytywanie sygnałów, wizualizację, wykonywanie ręcznego splotu oraz zapis wyników.

---

## Funkcje programu

- Wczytywanie wielu plików CSV zawierających dane sygnałowe (każda kolumna jako osobny kanał).
- Automatyczne generowanie wektora czasu (przyjęte próbkowanie: 1000 Hz).
- Wyświetlanie oscylogramu oraz widma częstotliwościowego (FFT).
- Ręczne wykonywanie splotu dwóch wybranych sygnałów.
- Zapis wynikowego sygnału do pliku CSV.
- Zapis aktualnie wyświetlanego wykresu do pliku PNG lub PDF.
- Intuicyjny interfejs graficzny zbudowany w `tkinter`.

---

## Wymagania

Program został napisany w Pythonie 3 i korzysta z następujących bibliotek:

- `tkinter` (część standardowej biblioteki Pythona)
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`

---

## Instalacja wymaganych pakietów

Wystarczy zainstalować potrzebne pakiety za pomocą `pip`:

```bash
pip install pandas numpy matplotlib seaborn scipy
```
Uwaga:
W niektórych systemach Linux może być konieczne wcześniejsze doinstalowanie tkinter (jeśli nie jest obecny w systemie):
```bash
sudo apt install python3-tk
```
## Uruchomienie programu
Po zainstalowaniu wymaganych pakietów, uruchom program bezpośrednio z terminala (bash):
```bash
python3 main.py
```
## Instrukcja obsługi
Szczegółowa instrukcja obsługi programu dostępna jest w menu: Pomoc → Instrukcja

## Obsługiwane formaty CSV
- Pliki z danymi sygnałów z automatycznym wykrywaniem formatu:

- Pierwsza kolumna jako czas: jeśli zawiera rosnące wartości większe lub równe zero — traktowana jako wektor czasu.

- Brak czasu: jeśli pierwsza kolumna nie spełnia warunku czasu — program generuje czas automatycznie (przyjęta częstotliwość próbkowania: 1000 Hz).

- Separatory: ; lub , (wykrywane automatycznie).

- Jeden lub więcej kanałów danych w kolejnych kolumnach.

- Brak nagłówków w plikach CSV (dane zaczynają się od pierwszego wiersza).

