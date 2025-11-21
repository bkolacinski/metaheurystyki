import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from genetic_algorithm import GeneticAlgorithm
from methods import *


def bar_plot(max_data: dict,
             avg_data: dict,
             title: str,
             xlabel: str,
             ylabel: str) -> None:
    labels = list(max_data.keys())
    max_values = list(max_data.values())
    avg_values = list(avg_data.values())

    x = np.arange(len(labels))

    _, ax = plt.subplots()

    bars_max = ax.bar(
        x,
        max_values,
        width=0.6,
        color='green',
        alpha=0.7,
        label='Max')

    bars_avg = ax.bar(
        x,
        avg_values,
        width=0.6,
        color='orange',
        alpha=0.8,
        label='Avg')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.legend()

    for _, (bar_max, max_val) in enumerate(zip(bars_max, max_values)):
        ax.annotate(f'{int(max_val)}',
                    xy=(bar_max.get_x() + bar_max.get_width() / 2, max_val),
                    xytext=(0, 1),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, color='black')

    for _, (bar_avg, avg_val) in enumerate(zip(bars_avg, avg_values)):
        ax.annotate(f'{int(avg_val)}',
                    xy=(bar_avg.get_x() + bar_avg.get_width() / 2, avg_val),
                    xytext=(0, 1),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, color='black')

    plt.tight_layout()
    plt.savefig(f"../plots/{title.replace(' ', '_')}.png")
    plt.close()


def analyze_csv_results(
        results_dir: str = "../results", verbose: bool = True) -> dict:
    results = {}

    csv_files = sorted(Path(results_dir).glob("*.csv"))

    for csv_file in csv_files:
        filename = csv_file.stem

        df = pd.read_csv(csv_file)

        best_fitness = df['best_fitness'].max()

        best_fitness_per_run = df.groupby(
            'run_id')['best_fitness'].max()
        avg_best_fitness = best_fitness_per_run.mean()

        results[filename] = {
            'best_fitness': best_fitness,
            'avg_best_fitness': avg_best_fitness
        }

        if verbose:
            print(f"\n{filename}:")
            print(f"  Best fitness: {best_fitness:,.0f}")
            print(f"  Avg best fitness: {avg_best_fitness:,.2f}")
            print(f"  Number of runs: {len(best_fitness_per_run)}")

    return results


def find_best_worst_by_prefix(results: dict) -> dict:
    tour_results = {
        k: v for k,
        v in results.items() if k.startswith('Tour')}
    roul_results = {
        k: v for k,
        v in results.items() if k.startswith('Roul')}

    tour_best = max(
        tour_results.items(),
        key=lambda x: x[1]['best_fitness'])
    tour_worst = min(
        tour_results.items(),
        key=lambda x: x[1]['best_fitness'])

    roul_best = max(
        roul_results.items(),
        key=lambda x: x[1]['best_fitness'])
    roul_worst = min(
        roul_results.items(),
        key=lambda x: x[1]['best_fitness'])

    selected = {
        'Tour_Best': (tour_best[0], tour_best[1]),
        'Tour_Worst': (tour_worst[0], tour_worst[1]),
        'Roul_Best': (roul_best[0], roul_best[1]),
        'Roul_Worst': (roul_worst[0], roul_worst[1])
    }

    print(f"\n\n{'=' * 80}")
    print("WYBRANE KONFIGURACJE DO WIZUALIZACJI")
    print(f"{'=' * 80}")

    for label, (config, data) in selected.items():
        print(f"\n{label}:")
        print(f"  Konfiguracja: {config}")
        print(f"  Best fitness: {data['best_fitness']:,.0f}")
        print(f"  Avg best fitness: {data['avg_best_fitness']:,.2f}")

    return selected


def plot_selected_configurations(selected: dict) -> None:
    max_data = {}
    avg_data = {}

    for label, (_, data) in selected.items():
        short_label = label
        max_data[short_label] = data['best_fitness']
        avg_data[short_label] = data['avg_best_fitness']

    os.makedirs('../plots', exist_ok=True)

    bar_plot(
        max_data=max_data,
        avg_data=avg_data,
        title="Porównanie najlepszych i najgorszych konfiguracji",
        xlabel="Konfiguracja",
        ylabel="Fitness"
    )


def main():
    results = analyze_csv_results("../results", verbose=True)

    print(f"\n\n{'=' * 60}")
    print(f"Podsumowanie: przeanalizowano {len(results)} plików")
    print(f"{'=' * 60}")

    selected = find_best_worst_by_prefix(results)

    plot_selected_configurations(selected)

    print(f"\n{'=' * 60}")
    print("Wykres zapisany w katalogu ../plots/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
