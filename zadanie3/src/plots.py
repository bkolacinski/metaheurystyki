import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


def bar_plot(max_data: dict,
             avg_data: dict,
             title: str,
             xlabel: str,
             ylabel: str,
             filename: str) -> None:
    labels = list(max_data.keys())
    max_values = list(max_data.values())
    avg_values = list(avg_data.values())

    labels_spread = np.arange(len(labels))

    _, ax = plt.subplots(figsize=(8, 5))

    bars_max = ax.bar(
        labels_spread,
        max_values,
        width=0.6,
        color='green',
        alpha=0.7,
        label='Max')

    bars_avg = ax.bar(
        labels_spread,
        avg_values,
        width=0.6,
        color='orange',
        alpha=0.8,
        label='Avg')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(labels_spread)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.legend()

    all_values = max_values + avg_values
    y_min, y_max = min(all_values), max(all_values)
    y_range = y_max - y_min

    if y_range > 0:
        log_y_min = np.log10(y_min)
        log_y_max = np.log10(y_max)
        log_range = log_y_max - log_y_min

        margin = log_range * 0.05

        ax.set_ylim(
            10 ** (log_y_min - margin),
            10 ** (log_y_max + margin)
        )

        log_ticks = np.linspace(log_y_min, log_y_max, 5)
        tick_values = [10 ** x for x in log_ticks]

        ax.yaxis.set_major_locator(
            ticker.FixedLocator(tick_values))
        ax.yaxis.set_minor_locator(ticker.NullLocator())

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f'{int(x):,}' if x >= 1 else f'{x:.2e}'
        ))

    ax.grid(True, which='major', alpha=0.3, axis='y')

    for bars, values in [(bars_max, max_values), (bars_avg, avg_values)]:
        for bar, val in zip(bars, values):
            ax.annotate(f'{int(val)}', ha='center', va='bottom',
                        xy=(bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 1), textcoords="offset points",
                        fontsize=8, color='black')

    plt.tight_layout()
    plt.savefig(f"../plots/{filename}")
    plt.close()


def plot_over_time(csv_filename: str,
                   filename: str,
                   config_name: str):
    df = pd.read_csv(f"../results/{csv_filename}.csv")

    best_fitness_per_run = df.groupby('run_id')['best_fitness'].max()
    best_run_id = best_fitness_per_run.idxmax()

    history_df = df[df['run_id'] ==
                    best_run_id].sort_values(by='iteration')

    history = []
    for _, row in history_df.iterrows():
        history.append({
            'iteration': int(row['iteration']),
            'best_fitness': row['best_fitness'],
            'avg_fitness': row['avg_fitness'],
            'worst_fitness': row['worst_fitness']
        })

    _, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))

    metrics = {
        0: {
            'data': [h['best_fitness'] for h in history],
            'label': 'Max Fitness',
            'title': 'Best Fitness',
            'color': 'green'
        },
        1: {
            'data': [h['avg_fitness'] for h in history],
            'label': 'Avg Fitness',
            'title': 'Average Fitness',
            'color': 'blue'
        },
        2: {
            'data': [h['worst_fitness'] for h in history],
            'label': 'Min Fitness',
            'title': 'Worst Fitness',
            'color': 'red'
        }
    }

    iterations = [h['iteration'] for h in history]

    for idx, metric in metrics.items():
        axes[idx].plot(iterations, metric['data'],
                       label=metric['label'], color=metric['color'])
        axes[idx].set_xlabel('Iteration')
        axes[idx].set_ylabel('Fitness')
        axes[idx].set_title(f"{metric['title']} - {config_name}")
        axes[idx].set_yscale('log')

        y_min, y_max = min(metric['data']), max(metric['data'])
        y_range = y_max - y_min

        if y_range > 0:
            log_y_min = np.log10(y_min)
            log_y_max = np.log10(y_max)
            log_range = log_y_max - log_y_min

            margin = log_range * 0.05

            axes[idx].set_ylim(
                10 ** (log_y_min - margin),
                10 ** (log_y_max + margin)
            )

            log_ticks = np.linspace(log_y_min, log_y_max, 5)
            tick_values = [10 ** x for x in log_ticks]

            axes[idx].yaxis.set_major_locator(
                ticker.FixedLocator(tick_values))
            axes[idx].yaxis.set_minor_locator(ticker.NullLocator())

            axes[idx].yaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, p: f'{int(x):,}' if x >= 1 else f'{x:.2e}'
            ))

        axes[idx].legend()
        axes[idx].grid(True, which='major', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"../plots/{filename}")
    plt.close()

    print(f"  Wykres zapisany: ../plots/{filename}")


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
    os.makedirs('../plots', exist_ok=True)

    max_data = {}
    avg_data = {}

    for label, (_, data_stats) in selected.items():
        short_label = label
        max_data[short_label] = data_stats['best_fitness']
        avg_data[short_label] = data_stats['avg_best_fitness']

    bar_plot(
        max_data=max_data,
        avg_data=avg_data,
        title="Porównanie najlepszych i najgorszych konfiguracji",
        xlabel="Konfiguracja",
        ylabel="Fitness",
        filename="comparison_bar_plot.png"
    )

    print(f"\nWykres słupkowy zapisany: ../plots/comparison_bar_plot.png")

    print("\n" + "=" * 80)
    print("GENEROWANIE WYKRESÓW OVER-TIME DLA WYBRANYCH KONFIGURACJI")
    print(f"{'=' * 80}")

    for label, (config_name, _) in selected.items():
        print(f"\nGenerowanie wykresu dla {label} ({config_name})...")

        plot_over_time(
            csv_filename=config_name,
            filename=f"{label}_over_time.png",
            config_name=config_name
        )


def main():
    print("=" * 80)
    print("ANALIZA WYNIKÓW Z PLIKÓW CSV")
    print("=" * 80)

    results = analyze_csv_results("../results", verbose=True)

    print(f"\n\n{'=' * 60}")
    print(f"Podsumowanie: przeanalizowano {len(results)} plików")
    print("=" * 60)

    selected = find_best_worst_by_prefix(results)

    plot_selected_configurations(selected)

    print("\n" + "=" * 60)
    print("ZAKOŃCZONO - Wszystkie wykresy zapisane w katalogu ../plots/")
    print("=" * 80)
    print("\nWygenerowane pliki:")
    print("  1. comparison_bar_plot.png - "
          "wykres słupkowy porównawczy")
    print("  2. Tour_Best_over_time.png - "
          "wykres over-time dla najlepszej konfiguracji Tournament")
    print("  3. Tour_Worst_over_time.png - "
          "wykres over-time dla najgorszej konfiguracji Tournament")
    print("  4. Roul_Best_over_time.png - "
          "wykres over-time dla najlepszej konfiguracji Roulette")
    print("  5. Roul_Worst_over_time.png - "
          "wykres over-time dla najgorszej konfiguracji Roulette")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
