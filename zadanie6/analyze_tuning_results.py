import argparse
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_tuning_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def plot_parameter_comparison(
    df: pd.DataFrame,
    param_name: str,
    instance_name: str,
    output_dir: str,
):
    param_values = sorted(df[param_name].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    vehicles_best = []
    vehicles_avg = []
    vehicles_std = []
    distances_best = []
    distances_avg = []
    distances_std = []

    for val in param_values:
        subset = df[df[param_name] == val]
        vehicles_best.append(subset["vehicles_best"].min())
        vehicles_avg.append(subset["vehicles_best"].mean())
        vehicles_std.append(subset["vehicles_best"].std())
        distances_best.append(subset["distance_best"].min())
        distances_avg.append(subset["distance_best"].mean())
        distances_std.append(subset["distance_best"].std())

    x_pos = np.arange(len(param_values))

    ax1.plot(
        x_pos,
        vehicles_best,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Najlepszy",
        color="green",
    )
    ax1.errorbar(
        x_pos,
        vehicles_avg,
        yerr=vehicles_std,
        marker="s",
        linewidth=2,
        markersize=6,
        label="Średni ± std",
        color="blue",
        capsize=5,
        alpha=0.7,
    )

    ax1.set_xlabel(
        f"Wartość parametru {param_name}", fontsize=12, fontweight="bold"
    )
    ax1.set_ylabel("Liczba pojazdów", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Wpływ parametru {param_name} na liczbę pojazdów\n{instance_name}",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{v}" for v in param_values])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(
        x_pos,
        distances_best,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Najlepszy",
        color="green",
    )
    ax2.errorbar(
        x_pos,
        distances_avg,
        yerr=distances_std,
        marker="s",
        linewidth=2,
        markersize=6,
        label="Średni ± std",
        color="blue",
        capsize=5,
        alpha=0.7,
    )

    ax2.set_xlabel(
        f"Wartość parametru {param_name}", fontsize=12, fontweight="bold"
    )
    ax2.set_ylabel("Całkowity dystans", fontsize=12, fontweight="bold")
    ax2.set_title(
        f"Wpływ parametru {param_name} na długość tras\n{instance_name}",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{v}" for v in param_values])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(
        output_dir, f"{instance_name}_param_{param_name}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Zapisano wykres: {output_path}")
    plt.close()


def plot_all_parameters(df: pd.DataFrame, instance_name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    parameters = ["alpha", "beta", "rho", "q0"]

    for param in parameters:
        if param in df.columns:
            plot_parameter_comparison(df, param, instance_name, output_dir)


def generate_summary_statistics(
    df: pd.DataFrame, instance_name: str, output_dir: str
):
    parameters = ["alpha", "beta", "rho", "q0"]

    summary_path = os.path.join(output_dir, f"{instance_name}_summary.txt")

    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"PODSUMOWANIE TUNINGU PARAMETRÓW: {instance_name}\n")
        f.write("=" * 70 + "\n\n")

        best_row = df.loc[df["distance_best"].idxmin()]
        f.write("NAJLEPSZA KONFIGURACJA:\n")
        f.write(f"  Pojazdy: {best_row['vehicles_best']}\n")
        f.write(f"  Dystans: {best_row['distance_best']:.2f}\n")
        f.write(f"  alpha: {best_row['alpha']}\n")
        f.write(f"  beta: {best_row['beta']}\n")
        f.write(f"  rho: {best_row['rho']}\n")
        f.write(f"  q0: {best_row['q0']}\n")
        f.write(f"  Czas: {best_row['time']:.2f}s\n\n")

        f.write("STATYSTYKI DLA KAŻDEGO PARAMETRU:\n")
        f.write("-" * 70 + "\n\n")

        for param in parameters:
            if param not in df.columns:
                continue

            f.write(f"Parametr: {param}\n")

            for val in sorted(df[param].unique()):
                subset = df[df[param] == val]
                avg_vehicles = subset["vehicles_best"].mean()
                min_vehicles = subset["vehicles_best"].min()
                avg_distance = subset["distance_best"].mean()
                min_distance = subset["distance_best"].min()

                f.write(f"  {param}={val}:\n")
                f.write(
                    f"    Pojazdy - Min: {min_vehicles}, Średnia: {avg_vehicles:.2f}\n"
                )
                f.write(
                    f"    Dystans - Min: {min_distance:.2f}, Średnia: {avg_distance:.2f}\n"
                )

            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("OGÓLNE STATYSTYKI:\n")
        f.write("=" * 70 + "\n")
        f.write(f"Liczba testowanych konfiguracji: {len(df)}\n")
        f.write(f"Minimalna liczba pojazdów: {df['vehicles_best'].min()}\n")
        f.write(f"Maksymalna liczba pojazdów: {df['vehicles_best'].max()}\n")
        f.write(f"Średnia liczba pojazdów: {df['vehicles_best'].mean():.2f}\n")
        f.write(f"Minimalny dystans: {df['distance_best'].min():.2f}\n")
        f.write(f"Maksymalny dystans: {df['distance_best'].max():.2f}\n")
        f.write(f"Średni dystans: {df['distance_best'].mean():.2f}\n")
        f.write(f"Całkowity czas tuningu: {df['time'].sum():.2f}s\n")

    print(f"Zapisano podsumowanie: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ACO parameter tuning results from CSV files"
    )
    parser.add_argument(
        "csv_file",
        help="Path to CSV file with tuning results",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis",
        help="Output directory for plots and analysis (default: analysis/)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"ERROR: CSV file not found: {args.csv_file}")
        sys.exit(1)

    print("=" * 70)
    print("ANALIZA WYNIKÓW TUNINGU PARAMETRÓW ACO")
    print("=" * 70)
    print(f"Plik wejściowy: {args.csv_file}")
    print(f"Katalog wyjściowy: {args.output_dir}")
    print("=" * 70)

    df = load_tuning_results(args.csv_file)

    instance_name = os.path.basename(args.csv_file).replace(".csv", "")

    print(f"\nZaładowano {len(df)} konfiguracji")
    print(
        f"Parametry: {', '.join([col for col in df.columns if col in ['alpha', 'beta', 'rho', 'q0']])}"
    )

    print("\nGenerowanie wykresów...")
    plot_all_parameters(df, instance_name, args.output_dir)

    print("\nGenerowanie statystyk...")
    generate_summary_statistics(df, instance_name, args.output_dir)

    print("\n" + "=" * 70)
    print("ANALIZA ZAKOŃCZONA")
    print("=" * 70)


if __name__ == "__main__":
    main()
