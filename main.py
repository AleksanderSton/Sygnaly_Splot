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

        # Dane
        self.data = None
        self.selected_signals = []
        self.oscillogram_data = None
        self.result_signal = None

        self.setup_ui()

    def setup_ui(self):
        # Główny frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Sekcja 1: Wczytanie plików
        file_frame = ttk.LabelFrame(main_frame, text="1. Wczytanie plików CSV", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(file_frame, text="Wczytaj pliki CSV",
                   command=self.load_csv_files).grid(row=0, column=0, padx=5)

        self.file_label = ttk.Label(file_frame, text="Nie wczytano plików")
        self.file_label.grid(row=0, column=1, padx=10)

        # Parametry generowania czasu
        params_frame = ttk.Frame(file_frame)
        params_frame.grid(row=1, column=0, columnspan=2, pady=5)

        ttk.Label(params_frame, text="Częstotliwość próbkowania (Hz):").grid(row=0, column=0, padx=5)
        self.fs_var = tk.StringVar(value="1000")
        ttk.Entry(params_frame, textvariable=self.fs_var, width=10).grid(row=0, column=1, padx=5)

        ttk.Label(params_frame, text="Czas startu (s):").grid(row=0, column=2, padx=5)
        self.t_start_var = tk.StringVar(value="0")
        ttk.Entry(params_frame, textvariable=self.t_start_var, width=10).grid(row=0, column=3, padx=5)

        # Sekcja 2: Wybór sygnałów
        signal_frame = ttk.LabelFrame(main_frame, text="2. Wybór sygnałów", padding="10")
        signal_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        self.signal_listbox = tk.Listbox(signal_frame, selectmode=tk.MULTIPLE, height=6)
        self.signal_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5)

        signal_buttons = ttk.Frame(signal_frame)
        signal_buttons.grid(row=0, column=1, padx=10)

        ttk.Button(signal_buttons, text="Wyświetl sygnały",
                   command=self.display_signals).grid(row=0, column=0, pady=2)
        ttk.Button(signal_buttons, text="Wykonaj splot ",
                   command=self.perform_manual_convolution).grid(row=1, column=0, pady=2)
        ttk.Button(signal_buttons, text="Wyświetl oscylogram",
                   command=self.display_oscillogram).grid(row=2, column=0, pady=2)
        ttk.Button(signal_buttons, text="Wyświetl wynikowy",
                   command=self.display_result_signal).grid(row=3, column=0, pady=2)

        # Sekcja 3: Wizualizacja
        viz_frame = ttk.LabelFrame(main_frame, text="3. Wizualizacja", padding="10")
        viz_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        # Matplotlib figure
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Sekcja 4: Zapis wyników
        save_frame = ttk.LabelFrame(main_frame, text="4. Zapis wyników", padding="10")
        save_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(save_frame, text="Zapisz wynikowy sygnał do CSV",
                   command=self.save_result_signal).grid(row=0, column=0, padx=5)
        ttk.Button(save_frame, text="Zapisz wykres",
                   command=self.save_plot).grid(row=0, column=1, padx=5)

        # Konfiguracja grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)

    def manual_convolution(self, x, h, dt_x, dt_h):
        """
        Ręczna implementacja splotu zgodnie ze wzorem:
        y(t) = ∫ x(τ)h(t-τ)dτ = ∫ x(t-τ)h(τ)dτ

        Parametry:
        x, h - sygnały wejściowe
        dt_x, dt_h - kroki czasowe sygnałów
        """

        # Długości sygnałów
        N_x = len(x)
        N_h = len(h)

        # Wybierz mniejszy krok czasowy dla lepszej dokładności
        dt = min(dt_x, dt_h)

        # Długość wyniku splotu (pełny splot)
        N_out = N_x + N_h - 1

        # Inicjalizacja wyniku
        y = np.zeros(N_out)

        # Implementacja wzoru splotu: y(n) = Σ x(k) * h(n-k)
        # gdzie n to indeks wyjściowy, k to indeks przesunięcia

        print(f"Rozpoczynam splot ręczny...")
        print(f"Długość x: {N_x}, Długość h: {N_h}")
        print(f"Długość wyniku: {N_out}")

        # Główna pętla splotu
        for n in range(N_out):
            # Dla każdej próbki wyjściowej
            sum_val = 0.0

            # Pętla po wszystkich możliwych przesunięciach τ
            for k in range(N_x):
                # Indeks dla h(n-k) -> h[n-k]
                h_idx = n - k

                # Sprawdź czy indeks jest w zakresie
                if 0 <= h_idx < N_h:
                    # Wykonaj mnożenie i dodaj do sumy
                    sum_val += x[k] * h[h_idx]

            y[n] = sum_val * dt  # Przybliżenie całki przez sumę z krokiem dt

            # Pokaż postęp co 10% obliczeń
            if (n + 1) % max(1, N_out // 10) == 0:
                progress = (n + 1) / N_out * 100
                print(f"Postęp: {progress:.1f}%")

        print("Splot ręczny zakończony!")

        return y

    def interpolate_signal(self, time, amplitude, new_dt):
        """Interpolacja sygnału do nowego kroku czasowego"""
        if len(time) < 2:
            return time, amplitude

        # Nowy wektor czasu
        t_start = time[0]
        t_end = time[-1]
        new_time = np.arange(t_start, t_end + new_dt, new_dt)

        # Interpolacja liniowa
        new_amplitude = np.interp(new_time, time, amplitude)

        return new_time, new_amplitude

    def detect_csv_format(self, file_path):
        """Wykryj format pliku CSV i określ strukturę danych"""
        try:
            # Próbuj wczytać pierwsze kilka linii
            with open(file_path, 'r', encoding='utf-8') as f:
                first_lines = [f.readline().strip() for _ in range(5)]

            # Sprawdź czy pierwsza linia zawiera nagłówki (literki)
            first_line = first_lines[0]
            has_header = any(char.isalpha() for char in first_line)

            # Spróbuj wczytać jako DataFrame
            if has_header:
                df = pd.read_csv(file_path)
            else:
                df = pd.read_csv(file_path, header=None)

            # Sprawdź strukturę danych
            n_cols = len(df.columns)
            n_rows = len(df)

            return {
                'has_header': has_header,
                'n_cols': n_cols,
                'n_rows': n_rows,
                'dataframe': df,
                'format_type': self._determine_format_type(df, has_header)
            }

        except Exception as e:
            raise Exception(f"Nie można przeanalizować pliku: {str(e)}")

    def _determine_format_type(self, df, has_header):
        """Określ typ formatu pliku CSV"""
        n_cols = len(df.columns)

        if n_cols == 1:
            return "single_column"  # Jedna kolumna z danymi sygnału
        elif n_cols == 2:
            return "time_signal"  # Dwie kolumny: czas, sygnał
        else:
            return "multi_signal"  # Wiele kolumn: czas + wiele sygnałów

    def load_csv_files(self):
        """Wczytanie plików CSV z sygnałami - obsługuje różne formaty"""
        file_paths = filedialog.askopenfilenames(
            title="Wybierz pliki CSV z sygnałami",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not file_paths:
            return

        try:
            all_data = {}
            fs = float(self.fs_var.get())  # Częstotliwość próbkowania
            t_start = float(self.t_start_var.get())  # Czas startu

            for file_path in file_paths:
                filename = os.path.basename(file_path)

                # Wykryj format pliku
                file_info = self.detect_csv_format(file_path)
                df = file_info['dataframe']
                format_type = file_info['format_type']

                if format_type == "single_column":
                    # Jedna kolumna - generuj czas automatycznie
                    if file_info['has_header']:
                        signal_data = df.iloc[:, 0].values
                        signal_name = f"{filename}_{df.columns[0]}"
                    else:
                        signal_data = df.iloc[:, 0].values
                        signal_name = f"{filename}_Signal"

                    # Generuj wektor czasu
                    n_samples = len(signal_data)
                    dt = 1.0 / fs
                    time_data = np.arange(n_samples) * dt + t_start

                    all_data[signal_name] = {
                        'time': time_data,
                        'amplitude': signal_data,
                        'file': filename,
                        'generated_time': True
                    }

                elif format_type == "time_signal":
                    # Dwie kolumny: czas i sygnał
                    time_data = df.iloc[:, 0].values
                    signal_data = df.iloc[:, 1].values

                    if file_info['has_header']:
                        signal_name = f"{filename}_{df.columns[1]}"
                    else:
                        signal_name = f"{filename}_Signal"

                    all_data[signal_name] = {
                        'time': time_data,
                        'amplitude': signal_data,
                        'file': filename,
                        'generated_time': False
                    }

                elif format_type == "multi_signal":
                    # Wiele kolumn: pierwsza to czas, pozostałe to sygnały
                    if file_info['has_header']:
                        time_col = df.columns[0]
                        signal_cols = df.columns[1:]
                        time_data = df[time_col].values

                        for col in signal_cols:
                            signal_name = f"{filename}_{col}"
                            all_data[signal_name] = {
                                'time': time_data,
                                'amplitude': df[col].values,
                                'file': filename,
                                'generated_time': False
                            }
                    else:
                        # Bez nagłówków - zakładaj pierwszą kolumnę jako czas
                        time_data = df.iloc[:, 0].values
                        for i in range(1, len(df.columns)):
                            signal_name = f"{filename}_Signal_{i}"
                            all_data[signal_name] = {
                                'time': time_data,
                                'amplitude': df.iloc[:, i].values,
                                'file': filename,
                                'generated_time': False
                            }

            if not all_data:
                messagebox.showerror("Błąd", "Nie udało się wczytać żadnych sygnałów")
                return

            self.data = all_data

            # Aktualizuj listę sygnałów
            self.signal_listbox.delete(0, tk.END)
            for signal_name in self.data.keys():
                self.signal_listbox.insert(tk.END, signal_name)

            # Pokaż informacje o wczytanych danych
            total_signals = len(self.data)
            generated_time_count = sum(1 for sig in self.data.values() if sig.get('generated_time', False))

            info_text = f"Wczytano {total_signals} sygnałów z {len(file_paths)} plików"
            if generated_time_count > 0:
                info_text += f"\n({generated_time_count} z wygenerowanym czasem)"

            self.file_label.config(text=info_text)

        except Exception as e:
            messagebox.showerror("Błąd", f"Błąd podczas wczytywania plików: {str(e)}")

    def get_selected_signals(self):
        """Pobierz wybrane sygnały"""
        selected_indices = self.signal_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Uwaga", "Nie wybrano żadnych sygnałów")
            return []

        selected_names = [list(self.data.keys())[i] for i in selected_indices]
        return selected_names

    def display_signals(self):
        """Wyświetlenie wybranych sygnałów"""
        selected_names = self.get_selected_signals()
        if not selected_names:
            return

        self.fig.clear()

        # Jeśli wybrano więcej niż 4 sygnały, użyj subplots
        n_signals = len(selected_names)
        if n_signals <= 4:
            for i, name in enumerate(selected_names):
                ax = self.fig.add_subplot(2, 2, i + 1) if n_signals > 1 else self.fig.add_subplot(1, 1, 1)
                signal_data = self.data[name]
                ax.plot(signal_data['time'], signal_data['amplitude'], linewidth=1.5)

                # Dodaj informację o wygenerowanym czasie
                title = name
                if signal_data.get('generated_time', False):
                    title += " (czas wygenerowany)"

                ax.set_title(title, fontsize=10)
                ax.set_xlabel('Czas [s]')
                ax.set_ylabel('Amplituda')
                ax.grid(True, alpha=0.3)
        else:
            # Wszystkie sygnały na jednym wykresie
            ax = self.fig.add_subplot(1, 1, 1)
            for name in selected_names:
                signal_data = self.data[name]
                label = name
                if signal_data.get('generated_time', False):
                    label += " (gen)"
                ax.plot(signal_data['time'], signal_data['amplitude'], label=label, linewidth=1.2)

            ax.set_title('Wybrane sygnały')
            ax.set_xlabel('Czas [s]')
            ax.set_ylabel('Amplituda')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()

    def perform_manual_convolution(self):
        """Wykonanie ręcznego splotu dwóch wybranych sygnałów"""
        selected_names = self.get_selected_signals()
        if len(selected_names) != 2:
            messagebox.showwarning("Uwaga", "Wybierz dokładnie 2 sygnały do splotu")
            return

        try:
            signal1 = self.data[selected_names[0]]
            signal2 = self.data[selected_names[1]]

            # Oblicz kroki czasowe
            dt1 = np.mean(np.diff(signal1['time'])) if len(signal1['time']) > 1 else 1.0 / float(self.fs_var.get())
            dt2 = np.mean(np.diff(signal2['time'])) if len(signal2['time']) > 1 else 1.0 / float(self.fs_var.get())

            print(f"Krok czasowy sygnału 1: {dt1:.6f} s")
            print(f"Krok czasowy sygnału 2: {dt2:.6f} s")

            # Wybierz wspólny krok czasowy (mniejszy dla lepszej dokładności)
            dt_common = min(dt1, dt2)
            print(f"Wspólny krok czasowy: {dt_common:.6f} s")

            # Interpoluj sygnały do wspólnego kroku czasowego jeśli potrzeba
            x1 = signal1['amplitude']
            x2 = signal2['amplitude']

            if abs(dt1 - dt_common) > 1e-10:
                print("Interpoluję sygnał 1...")
                _, x1 = self.interpolate_signal(signal1['time'], signal1['amplitude'], dt_common)

            if abs(dt2 - dt_common) > 1e-10:
                print("Interpoluję sygnał 2...")
                _, x2 = self.interpolate_signal(signal2['time'], signal2['amplitude'], dt_common)

            # Pokaż okno postępu
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Obliczanie splotu...")
            progress_window.geometry("300x100")
            progress_window.grab_set()

            progress_label = ttk.Label(progress_window, text="Rozpoczynam obliczenia splotu...")
            progress_label.pack(pady=20)

            progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
            progress_bar.pack(pady=10)
            progress_bar.start()

            # Aktualizuj GUI
            self.root.update()

            # Wykonaj ręczny splot
            print("Rozpoczynam ręczny splot...")
            convolution_result = self.manual_convolution(x1, x2, dt_common, dt_common)

            # Zamknij okno postępu
            progress_window.destroy()

            # Utwórz wektor czasu dla wyniku splotu
            n_result = len(convolution_result)
            time_start = signal1['time'][0] + signal2['time'][0]
            time_result = np.arange(n_result) * dt_common + time_start

            # Zapisz wynik
            self.result_signal = {
                'time': time_result,
                'amplitude': convolution_result,
                'name': f"Splot_ręczny_{selected_names[0]}_i_{selected_names[1]}"
            }

            # Oblicz statystyki
            max_val = np.max(convolution_result)
            min_val = np.min(convolution_result)
            mean_val = np.mean(convolution_result)

            messagebox.showinfo("Sukces",
                                f"Ręczny splot został wykonany pomyślnie!\n\n"
                                f"Długość wyniku: {len(convolution_result)} próbek\n")

        except Exception as e:
            # Zamknij okno postępu w przypadku błędu
            try:
                progress_window.destroy()
            except:
                pass
            messagebox.showerror("Błąd", f"Błąd podczas wykonywania ręcznego splotu: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def display_oscillogram(self):
        """Wyświetlenie oscylogramu dla wybranych sygnałów"""
        selected_names = self.get_selected_signals()
        if not selected_names:
            return

        self.fig.clear()

        if len(selected_names) == 1:
            # Pojedynczy oscylogram z analizą częstotliwościową
            signal_data = self.data[selected_names[0]]

            # Oscylogram
            ax1 = self.fig.add_subplot(2, 1, 1)
            ax1.plot(signal_data['time'], signal_data['amplitude'], 'b-', linewidth=1.5)

            title = f'Oscylogram - {selected_names[0]}'
            if signal_data.get('generated_time', False):
                title += " (czas wygenerowany)"
            ax1.set_title(title)
            ax1.set_xlabel('Czas [s]')
            ax1.set_ylabel('Amplituda')
            ax1.grid(True, alpha=0.3)

            # Analiza częstotliwościowa (FFT)
            ax2 = self.fig.add_subplot(2, 1, 2)
            n = len(signal_data['amplitude'])

            if n > 1:
                dt = np.mean(np.diff(signal_data['time']))

                # FFT
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
            # Wiele oscylogramów
            n_plots = len(selected_names)
            cols = 2 if n_plots > 2 else 1
            rows = (n_plots + cols - 1) // cols

            for i, name in enumerate(selected_names):
                ax = self.fig.add_subplot(rows, cols, i + 1)
                signal_data = self.data[name]
                ax.plot(signal_data['time'], signal_data['amplitude'], linewidth=1.2)

                title = name
                if signal_data.get('generated_time', False):
                    title += " (gen)"
                ax.set_title(title, fontsize=9)
                ax.set_xlabel('Czas [s]')
                ax.set_ylabel('Amplituda')
                ax.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()

    def display_result_signal(self):
        """Wyświetlenie sygnału wynikowego (po splocie)"""
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

        # Dodaj statystyki
        # amplitude = self.result_signal['amplitude']
        # stats_text = f"Max: {np.max(amplitude):.6f}, Min: {np.min(amplitude):.6f}, Średnia: {np.mean(amplitude):.6f}"
        # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        #         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        # Dodaj informację o metodzie
        # method_text = "Metoda: Ręczny splot (implementacja wzoru matematycznego)"
        # ax.text(0.02, 0.02, method_text, transform=ax.transAxes,
        #         verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

        self.fig.tight_layout()
        self.canvas.draw()

    def save_result_signal(self):
        """Zapisanie sygnału wynikowego do pliku CSV"""
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
                'Czas': self.result_signal['time'],
                'Amplituda': self.result_signal['amplitude']
            })

            # Dodaj metadane jako komentarz
            with open(file_path, 'w') as f:
                f.write(f"# Sygnał wynikowy: {self.result_signal['name']}\n")
                f.write(f"# Metoda: Ręczny splot (implementacja wzoru matematycznego)\n")
                f.write(f"# Data utworzenia: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Liczba próbek: {len(self.result_signal['amplitude'])}\n")
                f.write(f"# Max: {np.max(self.result_signal['amplitude']):.6f}\n")
                f.write(f"# Min: {np.min(self.result_signal['amplitude']):.6f}\n")
                f.write(f"# Średnia: {np.mean(self.result_signal['amplitude']):.6f}\n")

            df.to_csv(file_path, index=False, mode='a')
            messagebox.showinfo("Sukces", f"Sygnał zapisany do pliku: {file_path}")

        except Exception as e:
            messagebox.showerror("Błąd", f"Błąd podczas zapisywania: {str(e)}")

    def save_plot(self):
        """Zapisanie aktualnego wykresu"""
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
    """Funkcja główna programu"""
    root = tk.Tk()
    app = SignalAnalyzer(root)

    # Dodaj menu
    menubar = tk.Menu(root)
    root.config(menu=menubar)

    # Menu Plik
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Plik", menu=file_menu)
    file_menu.add_command(label="Wczytaj CSV", command=app.load_csv_files)
    file_menu.add_separator()
    file_menu.add_command(label="Zapisz sygnał", command=app.save_result_signal)
    file_menu.add_command(label="Zapisz wykres", command=app.save_plot)
    file_menu.add_separator()
    file_menu.add_command(label="Wyjście", command=root.quit)

    # Menu Pomoc
    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Pomoc", menu=help_menu)

    def show_help():
        help_text = """
Analizator Sygnałów - Instrukcja użytkowania:

OBSŁUGIWANE FORMATY CSV:
1. Jedna kolumna - tylko dane sygnału (czas generowany automatycznie)
2. Dwie kolumny - czas i sygnał
3. Wiele kolumn - czas + wiele sygnałów

PARAMETRY:
- Częstotliwość próbkowania: dla plików bez czasu (domyślnie 1000 Hz)
- Czas startu: początek generowanego czasu (domyślnie 0 s)

INSTRUKCJA:
1. Ustaw parametry częstotliwości próbkowania i czasu startu
2. Wczytaj pliki CSV z sygnałami
3. Wybierz sygnały z listy (Ctrl+klik dla wielu)
4. Użyj przycisków do:
   - Wyświetlenia sygnałów
   - Wykonania splotu (wybierz 2 sygnały)
   - Wyświetlenia oscylogramu z FFT
   - Wyświetlenia sygnału wynikowego
5. Zapisz wyniki do pliku CSV lub wykres

PRZYKŁADY PLIKÓW:
- signal.csv (jedna kolumna z danymi)
- time,signal.csv (czas i sygnał)  
- time,signal1,signal2.csv (czas i wiele sygnałów)
        """
        messagebox.showinfo("Pomoc", help_text)

    help_menu.add_command(label="Instrukcja", command=show_help)
    help_menu.add_command(label="O programie",
                          command=lambda: messagebox.showinfo("O programie",
                                                              "Analizator Sygnałów v2.0\n"
                                                              "Program do analizy sygnałów z plików CSV\n"
                                                              "Obsługuje różne formaty plików CSV\n"
                                                              "Automatyczne generowanie czasu dla plików bez kolumny czasowej"))

    root.mainloop()


if __name__ == "__main__":
    main()