from typing import Callable, List

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
        plt.savefig(filename)
        print(f"Wykres zapisano do: {filename}")
    else:
        plt.show()

    plt.close()
