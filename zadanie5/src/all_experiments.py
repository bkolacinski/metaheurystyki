import os
from typing import Any, Callable, Dict, List

import numpy as np
from experiments import (
    ExperimentStats,
    print_results_table,
    results_to_dataframe,
    run_multiple_experiments,
    run_parameter_study,
    save_results_to_csv,
)
from funcs import beale_function_batch, himmelblau_function_batch
from numpy.typing import NDArray
from plots import (
    plot_boxplot_comparison,
    plot_convergence_with_std,
    plot_heatmap,
    plot_parameter_influence,
)

N_RUNS = 15

BASE_PARAMS = {
    "swarm_size": 50,
    "iterations": 100,
    "w": 0.7,
    "c1": 1.5,
    "c2": 1.5,
    "use_randomness": True,
}

PARAM_RANGES = {
    "w": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "c1": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0],
    "c2": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0],
    "swarm_size": [5, 10, 20, 50, 100],
    "iterations": [5, 10, 20, 50, 100],
}

PARAM_LABELS = {
    "w": "Współczynnik inercji (w)",
    "c1": "Współczynnik kognitywny (c1)",
    "c2": "Współczynnik socjalny (c2)",
    "swarm_size": "Rozmiar roju",
    "iterations": "Liczba iteracji",
}


def setup_directories(func_name: str) -> tuple[str, str]:
    plots_dir = f"../plots/{func_name}"
    results_dir = f"../results/{func_name}"
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return plots_dir, results_dir


def run_all_parameter_studies(
    func: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    bounds: NDArray[np.floating],
    func_name: str,
    extrema: List[List[float]],
) -> Dict[str, Dict[Any, ExperimentStats]]:
    plots_dir, results_dir = setup_directories(func_name)

    all_results: Dict[str, Dict[Any, ExperimentStats]] = {}

    print(f"\n{'='*60}")
    print(f"EKSPERYMENTY DLA FUNKCJI: {func_name.upper()}")
    print(f"{'='*60}")

    for param_name, param_values in PARAM_RANGES.items():
        print(f"\n--- Badanie parametru: {param_name} ---")

        results = run_parameter_study(
            func=func,
            bounds=bounds,
            param_name=param_name,
            param_values=param_values,
            n_runs=N_RUNS,
            base_params=BASE_PARAMS.copy(),
            maximize=False,
        )

        all_results[param_name] = results

        df = results_to_dataframe(results, param_name)
        save_results_to_csv(
            df,
            f"{param_name}_study.csv",
            results_dir,
        )
        print_results_table(df, f"Wyniki badania parametru {param_name}")

        plot_parameter_influence(
            results=results,
            param_name=param_name,
            param_label=PARAM_LABELS[param_name],
            func_name=func_name,
            filename=f"{plots_dir}/influence_{param_name}.png",
        )

        plot_boxplot_comparison(
            results=results,
            param_name=param_name,
            param_label=PARAM_LABELS[param_name],
            func_name=func_name,
            filename=f"{plots_dir}/boxplot_{param_name}.png",
        )

    return all_results


def run_convergence_analysis(
    func: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    bounds: NDArray[np.floating],
    func_name: str,
) -> None:
    plots_dir, _ = setup_directories(func_name)

    print(f"\n--- Analiza zbieżności dla {func_name} ---")

    configs = [
        {"w": 0.3, "c1": 1.5, "c2": 1.5, "label": "w=0.3 (niska inercja)"},
        {"w": 0.7, "c1": 1.5, "c2": 1.5, "label": "w=0.7 (średnia inercja)"},
        {"w": 0.9, "c1": 1.5, "c2": 1.5, "label": "w=0.9 (wysoka inercja)"},
    ]

    all_histories: List[List[List[float]]] = []
    labels: List[str] = []

    for config in configs:
        label = config.pop("label")
        labels.append(label)

        params = BASE_PARAMS.copy()
        params.update(config)
        params["iterations"] = 200

        stats = run_multiple_experiments(
            func=func,
            bounds=bounds,
            n_runs=N_RUNS,
            maximize=False,
            **params,
        )

        all_histories.append(stats.all_convergence_histories)

    plot_convergence_with_std(
        all_histories=all_histories,
        labels=labels,
        title=f"Analiza zbieżności - {func_name}",
        filename=f"{plots_dir}/convergence_analysis.png",
    )


def run_best_configuration_visualization(
    func: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    bounds: NDArray[np.floating],
    func_name: str,
    extrema: List[List[float]],
) -> None:
    plots_dir, results_dir = setup_directories(func_name)

    print(f"\n--- Wizualizacja najlepszej konfiguracji dla {func_name} ---")

    best_params = {
        "swarm_size": 50,
        "iterations": 200,
        "w": 0.7,
        "c1": 1.5,
        "c2": 1.5,
        "use_randomness": True,
    }

    stats = run_multiple_experiments(
        func=func,
        bounds=bounds,
        n_runs=N_RUNS,
        maximize=False,
        **best_params,
    )

    best_idx = np.argmin(stats.all_values)
    best_position = stats.all_positions[best_idx]

    print("Najlepsza konfiguracja:")
    print(f"  Parametry: {best_params}")
    print(f"  Najlepszy wynik: {stats.best:.6f}")
    print(f"  Średni wynik: {stats.mean:.6f}")
    print(f"  Odch. std.: {stats.std:.6f}")
    print(f"  Najlepsza pozycja: {best_position}")

    summary = {
        "Parametr": list(best_params.keys()) + ["Najlepszy", "Średnia", "Odch. std."],
        "Wartość": list(best_params.values()) + [stats.best, stats.mean, stats.std],
    }

    import pandas as pd
    df_summary = pd.DataFrame(summary)
    save_results_to_csv(df_summary, "best_configuration_summary.csv", results_dir)

    plot_heatmap(
        func=func,
        bounds=bounds,
        extrema=extrema,
        found_position=best_position,
        title=f"{func_name} - Najlepsza znaleziona pozycja",
        filename=f"{plots_dir}/best_result_heatmap.png",
    )


def generate_summary_report(
    himmelblau_results: Dict[str, Dict[Any, ExperimentStats]],
    beale_results: Dict[str, Dict[Any, ExperimentStats]],
) -> None:
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("PODSUMOWANIE EKSPERYMENTÓW")
    print(f"{'='*60}")

    comparison_data = []

    for param_name in PARAM_RANGES.keys():
        h_results = himmelblau_results[param_name]
        b_results = beale_results[param_name]

        h_best_param = min(h_results.keys(), key=lambda k: h_results[k].mean)
        b_best_param = min(b_results.keys(), key=lambda k: b_results[k].mean)

        comparison_data.append({
            "Parametr": param_name,
            "Najlepsza wartość (Himmelblau)": h_best_param,
            "Średni wynik (Himmelblau)": h_results[h_best_param].mean,
            "Najlepsza wartość (Beale)": b_best_param,
            "Średni wynik (Beale)": b_results[b_best_param].mean,
        })

    import pandas as pd
    df_comparison = pd.DataFrame(comparison_data)
    save_results_to_csv(df_comparison, "comparison_summary.csv", results_dir)
    print_results_table(df_comparison, "Porównanie najlepszych parametrów")


def main():
    bounds_h = np.array([[-5, 5], [-5, 5]], dtype=float)
    extrema_h = [
        [3.0, 2.0],
        [-2.805118, 3.131312],
        [-3.779310, -3.283186],
        [3.584428, -1.848126],
    ]

    bounds_b = np.array([[-4.5, 4.5], [-4.5, 4.5]], dtype=float)
    extrema_b = [[3.0, 0.5]]

    himmelblau_results = run_all_parameter_studies(
        func=himmelblau_function_batch,
        bounds=bounds_h,
        func_name="Himmelblau",
        extrema=extrema_h,
    )

    run_convergence_analysis(
        func=himmelblau_function_batch,
        bounds=bounds_h,
        func_name="Himmelblau",
    )

    run_best_configuration_visualization(
        func=himmelblau_function_batch,
        bounds=bounds_h,
        func_name="Himmelblau",
        extrema=extrema_h,
    )

    beale_results = run_all_parameter_studies(
        func=beale_function_batch,
        bounds=bounds_b,
        func_name="Beale",
        extrema=extrema_b,
    )

    run_convergence_analysis(
        func=beale_function_batch,
        bounds=bounds_b,
        func_name="Beale",
    )

    run_best_configuration_visualization(
        func=beale_function_batch,
        bounds=bounds_b,
        func_name="Beale",
        extrema=extrema_b,
    )

    generate_summary_report(himmelblau_results, beale_results)

    print(f"\n{'='*60}")
    print("WSZYSTKIE EKSPERYMENTY ZAKOŃCZONE!")
    print(f"{'='*60}")
    print("\nWyniki zapisano w katalogach:")
    print("  - ../results/Himmelblau/")
    print("  - ../results/Beale/")
    print("  - ../results/")
    print("\nWykresy zapisano w katalogach:")
    print("  - ../plots/Himmelblau/")
    print("  - ../plots/Beale/")


if __name__ == "__main__":
    main()

