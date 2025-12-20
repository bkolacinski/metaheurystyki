import os

import numpy as np
from funcs import beale_function_batch, himmelblau_function_batch
from plots import plot_heatmap
from pso import PSO


def run_experiment(name, func, bounds, extrema, swarm_size=50, iterations=100):
    print(f"{'='*50}")
    print(f"Running Experiment: {name}")
    print(f"{'='*50}")

    pso = PSO(
        swarm_size=swarm_size,
        bounds=bounds,
        func=func,
        w=0.7,
        c1=1.49,
        c2=1.49,
        maximize=False,
        use_randomness=True,
    )

    output = pso.run(iterations=iterations)

    try:
        (best_pos, best_fit_internal), time_ms = output
    except ValueError:
        print("Error unpacking output.")
        return

    plots_path = "../plots"
    os.makedirs(plots_path, exist_ok=True)

    plot_heatmap(
        func=func,
        bounds=bounds,
        extrema=extrema,
        found_position=best_pos,
        title=f"{name} Function Landscape",
        filename=f"{plots_path}/{name.lower()}_heatmap.png",
    )

    real_fit = -best_fit_internal

    pos_str = ", ".join([f"{x: .4f}" for x in best_pos])

    print(f"Execution Time      : {time_ms:.2f} ms")
    print(f"Best Position Found : [{pos_str}]")
    print(f"Function Value      : {real_fit:.6f}")
    print(f"{'='*50}\n")


def main():
    bounds_h = np.array([[-5, 5], [-5, 5]], dtype=float)
    extrema_h = [
        [3.0, 2.0],
        [-2.805118, 3.131312],
        [-3.779310, -3.283186],
        [3.584428, -1.848126],
    ]
    run_experiment(
        "Himmelblau", himmelblau_function_batch, bounds_h, extrema_h
    )

    bounds_b = np.array([[-4.5, 4.5], [-4.5, 4.5]], dtype=float)
    extrema_b = [[3.0, 0.5]]
    run_experiment("Beale", beale_function_batch, bounds_b, extrema_b)


if __name__ == "__main__":
    main()
