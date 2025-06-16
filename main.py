import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
import os
from datetime import datetime


class SignalAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizator Sygnałów - Ręczny Splot")
        self.root.geometry("1200x800")

        self.data = None
        self.selected_signals = []
        self.oscillogram_data = None
        self.result_signal = None

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Sekcja wczytywania plików - uproszczona
        file_frame = ttk.LabelFrame(main_frame, text="1. Wczytanie plików CSV", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(file_frame, text="Wczytaj pliki CSV",
                   command=self.load_csv_files).grid(row=0, column=0, padx=5)

        self.file_label = ttk.Label(file_frame, text="Nie wczytano plików")
        self.file_label.grid(row=0, column=1, padx=10)

        # Sekcja wyboru sygnałów
        signal_frame = ttk.LabelFrame(main_frame, text="2. Wybór sygnałów", padding="10")
        signal_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        self.signal_listbox = tk.Listbox(signal_frame, selectmode=tk.MULTIPLE, height=6)
        self.signal_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5)

        signal_buttons = ttk.Frame(signal_frame)
        signal_buttons.grid(row=0, column=1, padx=10)

        ttk.Button(signal_buttons, text="Wyświetl oscylogram",
                   command=self.display_oscillogram).grid(row=0, column=0, pady=2)
        ttk.Button(signal_buttons, text="Wykonaj splot",
                   command=self.perform_manual_convolution).grid(row=1, column=0, pady=2)
        ttk.Button(signal_buttons, text="Wyświetl wynikowy",
                   command=self.display_result_signal).grid(row=2, column=0, pady=2)

        # Sekcja wizualizacji
        viz_frame = ttk.LabelFrame(main_frame, text="3. Wizualizacja", padding="10")
        viz_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Sekcja zapisu
        save_frame = ttk.LabelFrame(main_frame, text="4. Zapis wyników", padding="10")
        save_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(save_frame, text="Zapisz wynikowy sygnał do CSV",
                   command=self.save_result_signal).grid(row=0, column=0, padx=5)
        ttk.Button(save_frame, text="Zapisz wykres",
                   command=self.save_plot).grid(row=0, column=1, padx=5)

        # Konfiguracja responsywności
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)

    def manual_convolution(self, x, h, dt_x, dt_h):
        N_x = len(x)
        N_h = len(h)
        dt = min(dt_x, dt_h)
        N_out = N_x + N_h - 1
        y = np.zeros(N_out)

        print(f"Rozpoczynam splot ręczny...")
        print(f"Długość x: {N_x}, Długość h: {N_h}")
        print(f"Długość wyniku: {N_out}")

        for n in range(N_out):
            sum_val = 0.0
            for k in range(N_x):
                h_idx = n - k
                if 0 <= h_idx < N_h:
                    sum_val += x[k] * h[h_idx]
            y[n] = sum_val * dt

            if (n + 1) % max(1, N_out // 10) == 0:
                progress = (n + 1) / N_out * 100
                print(f"Postęp: {progress:.1f}%")

        print("Splot ręczny zakończony!")
        return y

    def interpolate_signal(self, time, amplitude, new_dt):
        if len(time) < 2:
            return time, amplitude

        t_start = time[0]
        t_end = time[-1]
        new_time = np.arange(t_start, t_end + new_dt, new_dt)
        new_amplitude = np.interp(new_time, time, amplitude)

        return new_time, new_amplitude

    def detect_csv_separator(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()

            if ';' in first_line:
                return ';'
            elif ',' in first_line:
                return ','
            else:
                return ';'
        except:
            return ';'

    def load_csv_files(self):
        file_paths = filedialog.askopenfilenames(
            title="Wybierz pliki CSV z sygnałami",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not file_paths:
            return

        try:
            all_data = {}

            for file_path in file_paths:
                filename = os.path.basename(file_path)

                separator = self.detect_csv_separator(file_path)
                df = pd.read_csv(file_path, sep=separator, header=None)

                n_cols = len(df.columns)
                n_samples = len(df)

                # Generujemy czas - wszystkie kolumny to kanały danych
                # Zakładamy próbkowanie 1000 Hz
                time_data = np.arange(n_samples) * 0.001

                # Dodajemy sygnały ze wszystkich kolumn
                for col_idx in range(n_cols):
                    signal_name = f"{filename}_Kanal_{col_idx + 1}"
                    signal_data = df.iloc[:, col_idx].values

                    all_data[signal_name] = {
                        'time': time_data,
                        'amplitude': signal_data,
                        'file': filename,
                        'channel': col_idx + 1
                    }

            if not all_data:
                messagebox.showerror("Błąd", "Nie udało się wczytać żadnych sygnałów")
                return

            self.data = all_data

            self.signal_listbox.delete(0, tk.END)
            for signal_name in self.data.keys():
                self.signal_listbox.insert(tk.END, signal_name)

            total_signals = len(self.data)
            total_files = len(file_paths)

            info_text = f"Wczytano {total_signals} sygnałów z {total_files} plików"
            self.file_label.config(text=info_text)

        except Exception as e:
            messagebox.showerror("Błąd", f"Błąd podczas wczytywania plików: {str(e)}")

    def get_selected_signals(self):
        selected_indices = self.signal_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Uwaga", "Nie wybrano żadnych sygnałów")
            return []

        selected_names = [list(self.data.keys())[i] for i in selected_indices]
        return selected_names

    def perform_manual_convolution(self):
        selected_names = self.get_selected_signals()
        if len(selected_names) != 2:
            messagebox.showwarning("Uwaga", "Wybierz dokładnie 2 sygnały do splotu")
            return

        try:
            signal1 = self.data[selected_names[0]]
            signal2 = self.data[selected_names[1]]

            dt1 = np.mean(np.diff(signal1['time'])) if len(signal1['time']) > 1 else 0.001
            dt2 = np.mean(np.diff(signal2['time'])) if len(signal2['time']) > 1 else 0.001

            print(f"Krok czasowy sygnału 1: {dt1:.6f} s")
            print(f"Krok czasowy sygnału 2: {dt2:.6f} s")

            dt_common = min(dt1, dt2)
            print(f"Wspólny krok czasowy: {dt_common:.6f} s")

            x1 = signal1['amplitude']
            x2 = signal2['amplitude']

            if abs(dt1 - dt_common) > 1e-10:
                print("Interpoluję sygnał 1...")
                _, x1 = self.interpolate_signal(signal1['time'], signal1['amplitude'], dt_common)

            if abs(dt2 - dt_common) > 1e-10:
                print("Interpoluję sygnał 2...")
                _, x2 = self.interpolate_signal(signal2['time'], signal2['amplitude'], dt_common)

            # Okno postępu
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Obliczanie splotu...")
            progress_window.geometry("300x100")
            progress_window.grab_set()

            progress_label = ttk.Label(progress_window, text="Rozpoczynam obliczenia splotu...")
            progress_label.pack(pady=20)

            progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
            progress_bar.pack(pady=10)
            progress_bar.start()

            self.root.update()

            print("Rozpoczynam ręczny splot...")
            convolution_result = self.manual_convolution(x1, x2, dt_common, dt_common)

            progress_window.destroy()

            # Poprawne obliczenie czasu dla wyniku splotu
            n_result = len(convolution_result)

            # Czas startowy to suma czasów startowych obu sygnałów
            time_start = signal1['time'][0] + signal2['time'][0]

            # Generujemy wektor czasu dla wyniku
            time_result = np.arange(n_result) * dt_common + time_start

            self.result_signal = {
                'time': time_result,
                'amplitude': convolution_result,
                'name': f"Splot_reczny_{selected_names[0]}_i_{selected_names[1]}",
                'dt': dt_common
            }

            messagebox.showinfo("Sukces",
                                f"Ręczny splot został wykonany pomyślnie!\n\n"
                                f"Długość wyniku: {len(convolution_result)} próbek\n"
                                f"Czas trwania: {(n_result - 1) * dt_common:.4f} s\n"
                                f"Krok czasowy: {dt_common:.6f} s")

        except Exception as e:
            try:
                progress_window.destroy()
            except:
                pass
            messagebox.showerror("Błąd", f"Błąd podczas wykonywania ręcznego splotu: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def display_oscillogram(self):
        selected_names = self.get_selected_signals()
        if not selected_names:
            return

        self.fig.clear()

        if len(selected_names) == 1:
            signal_data = self.data[selected_names[0]]

            # Oscylogram
            ax1 = self.fig.add_subplot(2, 1, 1)
            ax1.plot(signal_data['time'], signal_data['amplitude'], 'b-', linewidth=1.5)
            ax1.set_title(f'Oscylogram - {selected_names[0]}')
            ax1.set_xlabel('Czas [s]')
            ax1.set_ylabel('Amplituda')
            ax1.grid(True, alpha=0.3)

            # Widmo częstotliwościowe
            ax2 = self.fig.add_subplot(2, 1, 2)
            n = len(signal_data['amplitude'])

            if n > 1:
                dt = np.mean(np.diff(signal_data['time']))
                yf = fft(signal_data['amplitude'])
                xf = fftfreq(n, dt)[:n // 2]

                ax2.plot(xf, 2.0 / n * np.abs(yf[0:n // 2]))
                ax2.set_title('Widmo częstotliwościowe')
                ax2.set_xlabel('Częstotliwość [Hz]')
                ax2.set_ylabel('Amplituda')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Za mało danych dla FFT',
                         horizontalalignment='center', verticalalignment='center',
                         transform=ax2.transAxes)

        else:
            # Wyświetl wszystkie wybrane sygnały
            n_plots = len(selected_names)
            cols = 2 if n_plots > 2 else 1
            rows = (n_plots + cols - 1) // cols

            for i, name in enumerate(selected_names):
                ax = self.fig.add_subplot(rows, cols, i + 1)
                signal_data = self.data[name]
                ax.plot(signal_data['time'], signal_data['amplitude'], linewidth=1.2)
                ax.set_title(name, fontsize=9)
                ax.set_xlabel('Czas [s]')
                ax.set_ylabel('Amplituda')
                ax.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()

    def display_result_signal(self):
        if self.result_signal is None:
            messagebox.showwarning("Uwaga", "Najpierw wykonaj splot sygnałów")
            return

        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)

        ax.plot(self.result_signal['time'], self.result_signal['amplitude'], 'r-', linewidth=1.5)
        ax.set_title(f"Sygnał wynikowy: {self.result_signal['name']}")
        ax.set_xlabel('Czas [s]')
        ax.set_ylabel('Amplituda')
        ax.grid(True, alpha=0.3)

        # Dodaj informacje o zakresie czasowym
        t_start = self.result_signal['time'][0]
        t_end = self.result_signal['time'][-1]
        ax.text(0.02, 0.98, f'Zakres czasu: {t_start:.4f} - {t_end:.4f} s',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        self.fig.tight_layout()
        self.canvas.draw()

    def save_result_signal(self):
        if self.result_signal is None:
            messagebox.showwarning("Uwaga", "Brak sygnału wynikowego do zapisania")
            return

        file_path = filedialog.asksaveasfilename(
            title="Zapisz sygnał wynikowy",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            df = pd.DataFrame({
                'time': self.result_signal['time'],
                'amplitude': self.result_signal['amplitude']
            })

            df.to_csv(file_path, index=False, header=False, sep=';')
            messagebox.showinfo("Sukces", f"Sygnał zapisany do pliku: {file_path}")

        except Exception as e:
            messagebox.showerror("Błąd", f"Błąd podczas zapisywania: {str(e)}")

    def save_plot(self):
        file_path = filedialog.asksaveasfilename(
            title="Zapisz wykres",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Sukces", f"Wykres zapisany do pliku: {file_path}")

        except Exception as e:
            messagebox.showerror("Błąd", f"Błąd podczas zapisywania wykresu: {str(e)}")


def main():
    root = tk.Tk()
    app = SignalAnalyzer(root)

    # Menu
    menubar = tk.Menu(root)
    root.config(menu=menubar)

    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Plik", menu=file_menu)
    file_menu.add_command(label="Wczytaj CSV", command=app.load_csv_files)
    file_menu.add_separator()
    file_menu.add_command(label="Zapisz sygnał", command=app.save_result_signal)
    file_menu.add_command(label="Zapisz wykres", command=app.save_plot)
    file_menu.add_separator()
    file_menu.add_command(label="Wyjście", command=root.quit)

    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Pomoc", menu=help_menu)

    def show_help():
        help_text = """
Analizator Sygnałów - Instrukcja użytkowania:

OBSŁUGIWANE FORMATY CSV:
- Pliki z danymi sygnałów bez nagłówków
- Wszystkie kolumny zawierają kanały danych (nie czas)
- Czas generowany automatycznie z częstotliwością 1000 Hz
- Separatory: ; lub , (automatyczne wykrywanie)
- Jeden lub więcej kanałów danych w kolejnych kolumnach

INSTRUKCJA:
1. Wczytaj pliki CSV z sygnałami
2. Wybierz sygnały z listy (Ctrl+klik dla wielu)
3. Użyj przycisków do:
   - Wyświetlenia oscylogramu z FFT
   - Wykonania splotu (wybierz dokładnie 2 sygnały)
   - Wyświetlenia sygnału wynikowego
4. Zapisz wyniki do pliku CSV lub wykres

UWAGI:
- Splot oblicza poprawny czas wynikowy na podstawie czasów oryginalnych sygnałów
- Program automatycznie interpoluje sygnały do wspólnego kroku czasowego
- Wynik splotu zawiera informacje o zakresie czasowym
- Wszystkie kanały traktowane są jako dane sygnałów
        """
        messagebox.showinfo("Pomoc", help_text)

    help_menu.add_command(label="Instrukcja", command=show_help)
    help_menu.add_command(label="O programie",
                          command=lambda: messagebox.showinfo("O programie",
                                                              "Analizator Sygnałów v1.0\n"
                                                              "Program do analizy sygnałów z plików CSV\n"
                                                              "Automatyczne generowanie czasu (1000 Hz)\n"
                                                              "Wszystkie kolumny to kanały danych"))

    root.mainloop()


if __name__ == "__main__":
    main()