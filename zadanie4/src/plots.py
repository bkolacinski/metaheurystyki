import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_route(
    data: list[tuple[int, int, int]],
    path: list[int],
    index_map: dict,
    title: str,
    filename: str,
    plots_dir: str
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))

    x_coords = [d[1] for d in data]
    y_coords = [d[2] for d in data]

    ax.scatter(x_coords, y_coords, c='blue', s=100, zorder=5, label='Atrakcje')

    for i, (id_num, x, y) in enumerate(data):
        ax.annotate(
            str(id_num), (x, y),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8, fontweight='bold'
        )

    if path:
        route_x = [data[idx][1] for idx in path]
        route_y = [data[idx][2] for idx in path]

        route_ids = [index_map[idx] for idx in path]
        print(f"  Trasa: {' -> '.join(map(str, route_ids))}")

        ax.plot(
            route_x, route_y, 'r-', linewidth=2, alpha=0.7,
            label='Trasa', zorder=3
        )

        ax.scatter(
            [route_x[0]], [route_y[0]], c='lime', s=300,
            marker='*', zorder=7, label='Start', edgecolors='darkgreen', linewidths=2
        )

        ax.scatter(
            [route_x[-1]], [route_y[-1]], c='red', s=200,
            marker='s', zorder=7, label='Koniec', edgecolors='darkred', linewidths=2
        )

    ax.set_xlabel('Współrzędna X', fontsize=12)
    ax.set_ylabel('Współrzędna Y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"  Wykres trasy zapisany: {filepath}")


def plot_convergence(
    histories: list[list[float]],
    title: str,
    filename: str,
    plots_dir: str
) -> None:
    valid_histories = [h for h in histories if h]
    if not valid_histories:
        print("  Warning: No valid histories to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    max_len = max(len(h) for h in valid_histories)
    histories_padded = []
    for h in valid_histories:
        padded = h + [h[-1]] * (max_len - len(h))
        histories_padded.append(padded)

    histories_array = np.array(histories_padded)

    iterations = np.arange(max_len)
    mean_values = np.mean(histories_array, axis=0)
    min_values = np.min(histories_array, axis=0)
    max_values = np.max(histories_array, axis=0)

    ax.plot(iterations, mean_values, 'b-', linewidth=2, label='Średnia')

    ax.fill_between(
        iterations, min_values, max_values,
        alpha=0.3, color='blue', label='Zakres min-max'
    )

    ax.set_xlabel('Iteracja', fontsize=12)
    ax.set_ylabel('Najlepszy koszt (długość trasy)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"  Wykres zbieżności zapisany: {filepath}")


def plot_parameter_comparison(
    data: list[dict],
    param_name: str,
    param_label: str,
    title: str,
    filename: str,
    plots_dir: str
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    param_values = sorted(set(d['varied_value'] for d in data))
    boxplot_data = []

    for val in param_values:
        matching = [d for d in data if d['varied_value'] == val]
        costs = []
        for m in matching:
            costs.extend(m['all_costs'])
        boxplot_data.append(costs)

    ax.boxplot(boxplot_data, tick_labels=[str(v) for v in param_values])
    ax.set_xlabel(param_label, fontsize=14)
    ax.set_ylabel('Długość trasy', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"  Wykres porównawczy zapisany: {filepath}")


def plot_bar_comparison(
    labels: list[str],
    mean_values: list[float],
    min_values: list[float],
    max_values: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    plots_dir: str
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(labels))
    errors = [
        [m - mi for m, mi in zip(mean_values, min_values)],
        [ma - m for m, ma in zip(mean_values, max_values)]
    ]

    bars = ax.bar(x, mean_values, yerr=errors, capsize=5, color='steelblue', alpha=0.8)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"  Wykres słupkowy zapisany: {filepath}")


def generate_best_route_plots(data_dir: str, results_dir: str, plots_dir: str) -> None:
    import ast
    from read_data import read_data

    os.makedirs(plots_dir, exist_ok=True)

    for data_name in ['A-n32-k5', 'A-n80-k10']:
        data_file = os.path.join(data_dir, f'{data_name}.txt')
        summary_file = os.path.join(results_dir, data_name, 'summary.csv')

        if not os.path.exists(data_file):
            print(f"Brak pliku danych: {data_file}")
            continue
        if not os.path.exists(summary_file):
            print(f"Brak pliku wyników: {summary_file}")
            continue

        print(f"\n{'=' * 60}")
        print(f"GENEROWANIE WYKRESU NAJLEPSZEJ TRASY DLA: {data_name}")
        print(f"{'=' * 60}")

        data, distances_matrix, index_map = read_data(data_file)

        summary_df = pd.read_csv(summary_file)

        best_row = summary_df.loc[summary_df['best_cost'].idxmin()]
        best_cost = best_row['best_cost']
        best_path = ast.literal_eval(best_row['best_path'])

        print(f"  Najlepszy koszt: {best_cost:.2f}")
        print(f"  Parametry: {best_row['params']}")

        data_plots_dir = os.path.join(plots_dir, data_name)
        os.makedirs(data_plots_dir, exist_ok=True)

        plot_route(
            data=data,
            path=best_path,
            index_map=index_map,
            title=f'Najlepsza trasa dla {data_name} (koszt: {best_cost:.2f})',
            filename='best_route.png',
            plots_dir=data_plots_dir
        )


def analyze_and_plot_results(results_dir: str, plots_dir: str) -> None:
    os.makedirs(plots_dir, exist_ok=True)

    for data_name in ['A-n32-k5', 'A-n80-k10']:
        data_results_dir = os.path.join(results_dir, data_name)
        data_plots_dir = os.path.join(plots_dir, data_name)
        os.makedirs(data_plots_dir, exist_ok=True)

        summary_file = os.path.join(data_results_dir, 'summary.csv')
        if not os.path.exists(summary_file):
            print(f"Brak pliku summary.csv dla {data_name}")
            continue

        print(f"\n{'=' * 60}")
        print(f"GENEROWANIE WYKRESÓW DLA: {data_name}")
        print(f"{'=' * 60}")

        summary_df = pd.read_csv(summary_file)

        experiment_configs = {
            'm': ('Liczba mrówek (m)', 'Wpływ liczby mrówek na wyniki'),
            'p_random': ('Prawdopodobieństwo losowego wyboru (p_random)',
                        'Wpływ prawdopodobieństwa losowego wyboru'),
            'alpha': ('Wpływ feromonów (α)', 'Wpływ współczynnika α na wyniki'),
            'beta': ('Wpływ heurystyki (β)', 'Wpływ współczynnika β na wyniki'),
            'iterations': ('Liczba iteracji', 'Wpływ liczby iteracji na wyniki'),
            'rho': ('Współczynnik wyparowywania (ρ)',
                   'Wpływ współczynnika wyparowywania na wyniki'),
        }

        for exp_type, (xlabel, title) in experiment_configs.items():
            exp_data = summary_df[summary_df['experiment'] == exp_type]
            if exp_data.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 7))

            param_values = sorted(exp_data['varied_value'].unique())

            means = exp_data.sort_values('varied_value')['mean'].values
            mins = exp_data.sort_values('varied_value')['min'].values
            maxs = exp_data.sort_values('varied_value')['max'].values

            x = np.arange(len(param_values))
            errors = [means - mins, maxs - means]

            ax.bar(x, means, yerr=errors, capsize=5, color='steelblue', alpha=0.8)
            ax.set_xlabel(xlabel, fontsize=14)
            ax.set_ylabel('Długość trasy', fontsize=14)
            ax.set_title(f'{title} ({data_name})', fontsize=16, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in param_values])
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            filepath = os.path.join(data_plots_dir, f'influence_{exp_type}.png')
            plt.savefig(filepath, dpi=300)
            plt.close()
            print(f"  Wykres zapisany: {filepath}")

        print(f"\n{'=' * 60}")
        print(f"STATYSTYKI DLA: {data_name}")
        print(f"{'=' * 60}")

        for exp_type in experiment_configs.keys():
            exp_data = summary_df[summary_df['experiment'] == exp_type]
            if exp_data.empty:
                continue

            print(f"\n{experiment_configs[exp_type][0]}:")
            for _, row in exp_data.sort_values('varied_value').iterrows():
                print(f"  {row['varied_value']}: "
                      f"średnia={row['mean']:.2f}, "
                      f"mediana={row['median']:.2f}, "
                      f"min={row['min']:.2f}, "
                      f"max={row['max']:.2f}, "
                      f"std={row['std']:.2f}")


def main():
    print("=" * 80)
    print("GENEROWANIE WYKRESÓW DLA ALGORYTMU MRÓWKOWEGO")
    print("=" * 80)

    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, '..', 'results')
    plots_dir = os.path.join(base_dir, '..', 'plots')
    data_dir = os.path.join(base_dir, '..', 'data')

    generate_best_route_plots(data_dir, results_dir, plots_dir)

    analyze_and_plot_results(results_dir, plots_dir)

    print("\n" + "=" * 80)
    print("ZAKOŃCZONO - Wykresy zapisane w katalogu plots/")
    print("=" * 80)


if __name__ == "__main__":
    main()
