from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from numpy.typing import NDArray


def plot_heatmap(
    func: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    bounds: NDArray[np.floating],
    extrema: List[NDArray[np.floating]] | NDArray[np.floating] | None = None,
    found_position: NDArray[np.floating] | None = None,
    title: str = "Function Heatmap",
    filename: str | None = None,
) -> None:
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    resolution = 200
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    Z = func(grid_points).reshape(X.shape)

    plt.figure(figsize=(10, 8))

    heatmap = plt.pcolormesh(
        X,
        Y,
        Z,
        cmap="viridis",
        norm=LogNorm(vmin=Z.min(), vmax=Z.max()),
        shading="gouraud",
    )

    plt.colorbar(heatmap, label="Wartość funkcji (skala logarytmiczna)")

    if extrema is not None:
        if isinstance(extrema, list):
            extrema_arr = np.array(extrema)
        else:
            extrema_arr = extrema

        plt.scatter(
            extrema_arr[:, 0],
            extrema_arr[:, 1],
            color="red",
            marker="x",
            s=100,
            label="Prawdziwe ekstrema",
        )

    if found_position is not None:
        plt.scatter(
            found_position[0],
            found_position[1],
            color="white",
            edgecolors="black",
            marker="*",
            s=150,
            label="Znalezione rozwiązanie",
        )

    if extrema is not None or found_position is not None:
        plt.legend()

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    x_ticks = np.arange(np.ceil(x_min), np.floor(x_max) + 1, 1.0)
    y_ticks = np.arange(np.ceil(y_min), np.floor(y_max) + 1, 1.0)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Wykres zapisano do: {filename}")
    else:
        plt.show()

    plt.close()


def plot_parameter_influence(
    results: Dict[Any, Any],
    param_name: str,
    param_label: str,
    func_name: str,
    filename: str | None = None,
) -> None:
    param_values = list(results.keys())
    means = [results[v].mean for v in param_values]
    stds = [results[v].std for v in param_values]
    bests = [results[v].best for v in param_values]
    worsts = [results[v].worst for v in param_values]

    epsilon = 1e-10
    means_shifted = [max(v, epsilon) for v in means]
    bests_shifted = [max(v, epsilon) for v in bests]
    worsts_shifted = [max(v, epsilon) for v in worsts]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    x_pos = np.arange(len(param_values))

    ax1.bar(x_pos, means_shifted, capsize=5, color='steelblue',
            edgecolor='black', alpha=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(v) for v in param_values])
    ax1.set_xlabel(param_label)
    ax1.set_ylabel('Wartość funkcji celu (skala log)')
    ax1.set_title(f'Średnia dla różnych wartości {param_name}')
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3)

    ax2 = axes[1]
    width = 0.25

    ax2.bar(x_pos - width, bests_shifted, width, label='Najlepszy', color='green', alpha=0.8)
    ax2.bar(x_pos, means_shifted, width, label='Średnia', color='steelblue', alpha=0.8)
    ax2.bar(x_pos + width, worsts_shifted, width, label='Najgorszy', color='red', alpha=0.8)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(v) for v in param_values])
    ax2.set_xlabel(param_label)
    ax2.set_ylabel('Wartość funkcji celu (skala log)')
    ax2.set_title(f'Porównanie wyników dla różnych wartości {param_name}')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Wpływ parametru {param_name} - funkcja {func_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Wykres zapisano do: {filename}")
    else:
        plt.show()

    plt.close()


def plot_convergence_comparison(
    convergence_histories: List[List[float]],
    labels: List[str],
    title: str = "Porównanie zbieżności",
    filename: str | None = None,
) -> None:
    plt.figure(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(convergence_histories)))

    for history, label, color in zip(convergence_histories, labels, colors):
        iterations = range(len(history))
        plt.plot(iterations, history, label=label, color=color, linewidth=1.5)

    plt.xlabel('Iteracja')
    plt.ylabel('Wartość funkcji celu')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.yscale('log')

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Wykres zapisano do: {filename}")
    else:
        plt.show()

    plt.close()


def plot_convergence_with_std(
    all_histories: List[List[List[float]]],
    labels: List[str],
    title: str = "Zbieżność algorytmu",
    filename: str | None = None,
    max_iterations: int = 50,
) -> None:
    plt.figure(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_histories)))

    for histories, label, color in zip(all_histories, labels, colors):
        min_len = min(len(h) for h in histories)
        truncated = [h[:min_len] for h in histories]

        arr = np.array(truncated)
        mean_history = np.mean(arr, axis=0)
        std_history = np.std(arr, axis=0)

        max_len = min(len(mean_history), max_iterations)
        mean_history = mean_history[:max_len]
        std_history = std_history[:max_len]

        iterations = range(len(mean_history))

        plt.plot(iterations, mean_history, label=label, color=color, linewidth=2)
        plt.fill_between(
            iterations,
            mean_history - std_history,
            mean_history + std_history,
            color=color,
            alpha=0.2,
        )

    plt.xlabel('Iteracja')
    plt.ylabel('Wartość funkcji celu')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.xlim(0, max_iterations)

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Wykres zapisano do: {filename}")
    else:
        plt.show()

    plt.close()


def plot_boxplot_comparison(
    results: Dict[Any, Any],
    param_name: str,
    param_label: str,
    func_name: str,
    filename: str | None = None,
) -> None:
    plt.figure(figsize=(10, 6))

    data = [results[v].all_values for v in results.keys()]
    labels = [str(v) for v in results.keys()]

    epsilon = 1e-10
    data_shifted = [[max(v, epsilon) for v in d] for d in data]

    bp = plt.boxplot(data_shifted, labels=labels, patch_artist=True)

    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.xlabel(param_label)
    plt.ylabel('Wartość funkcji celu (skala log)')
    plt.title(f'Rozkład wyników dla różnych wartości {param_name} - {func_name}')
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.3)

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Wykres zapisano do: {filename}")
    else:
        plt.show()

    plt.close()
