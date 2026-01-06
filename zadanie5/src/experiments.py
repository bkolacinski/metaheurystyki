import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pso import PSO


@dataclass
class ExperimentResult:
    best_position: NDArray[np.floating]
    best_value: float
    convergence_history: List[float]
    execution_time_ms: float


@dataclass
class ExperimentStats:
    best: float
    worst: float
    mean: float
    median: float
    std: float
    all_values: List[float]
    all_positions: List[NDArray[np.floating]]
    all_convergence_histories: List[List[float]]
    all_execution_times: List[float]


def run_single_experiment(
    func: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    bounds: NDArray[np.floating],
    swarm_size: int = 30,
    iterations: int = 100,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    maximize: bool = False,
    use_randomness: bool = True,
) -> ExperimentResult:
    pso = PSO(
        swarm_size=swarm_size,
        bounds=bounds,
        func=func,
        w=w,
        c1=c1,
        c2=c2,
        maximize=maximize,
        use_randomness=use_randomness,
    )

    _, time_ms = pso.run(iterations=iterations)
    best_pos, best_val = pso.get_best_result()
    convergence = pso.get_convergence_history()

    return ExperimentResult(
        best_position=best_pos,
        best_value=best_val,
        convergence_history=convergence,
        execution_time_ms=time_ms,
    )


def run_multiple_experiments(
    func: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    bounds: NDArray[np.floating],
    n_runs: int = 5,
    swarm_size: int = 30,
    iterations: int = 100,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    maximize: bool = False,
    use_randomness: bool = True,
) -> ExperimentStats:
    results: List[ExperimentResult] = []

    for _ in range(n_runs):
        result = run_single_experiment(
            func=func,
            bounds=bounds,
            swarm_size=swarm_size,
            iterations=iterations,
            w=w,
            c1=c1,
            c2=c2,
            maximize=maximize,
            use_randomness=use_randomness,
        )
        results.append(result)

    all_values = [r.best_value for r in results]
    all_positions = [r.best_position for r in results]
    all_convergence = [r.convergence_history for r in results]
    all_times = [r.execution_time_ms for r in results]

    return ExperimentStats(
        best=min(all_values) if not maximize else max(all_values),
        worst=max(all_values) if not maximize else min(all_values),
        mean=float(np.mean(all_values)),
        median=float(np.median(all_values)),
        std=float(np.std(all_values)),
        all_values=all_values,
        all_positions=all_positions,
        all_convergence_histories=all_convergence,
        all_execution_times=all_times,
    )


def run_parameter_study(
    func: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    bounds: NDArray[np.floating],
    param_name: str,
    param_values: List[Any],
    n_runs: int = 5,
    base_params: Dict[str, Any] | None = None,
    maximize: bool = False,
) -> Dict[Any, ExperimentStats]:
    if base_params is None:
        base_params = {
            "swarm_size": 30,
            "iterations": 100,
            "w": 0.7,
            "c1": 1.5,
            "c2": 1.5,
            "use_randomness": True,
        }

    results: Dict[Any, ExperimentStats] = {}

    for value in param_values:
        params = base_params.copy()
        params[param_name] = value

        print(f"  Badanie {param_name}={value}...")

        stats = run_multiple_experiments(
            func=func,
            bounds=bounds,
            n_runs=n_runs,
            maximize=maximize,
            **params,
        )
        results[value] = stats

    return results


def results_to_dataframe(
    results: Dict[Any, ExperimentStats],
    param_name: str,
) -> pd.DataFrame:
    data = []
    for param_value, stats in results.items():
        data.append({
            param_name: param_value,
            "Najlepszy": stats.best,
            "Najgorszy": stats.worst,
            "Średnia": stats.mean,
            "Mediana": stats.median,
            "Odch. std.": stats.std,
            "Śr. czas [ms]": np.mean(stats.all_execution_times),
        })

    return pd.DataFrame(data)


def save_results_to_csv(
    df: pd.DataFrame,
    filename: str,
    results_dir: str = "../results",
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    df.to_csv(filepath, index=False, sep=";", decimal=",")
    print(f"Wyniki zapisano do: {filepath}")


def print_results_table(df: pd.DataFrame, title: str = "") -> None:
    if title:
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")

    pd.set_option('display.float_format', lambda x: f'{x:.6f}')
    print(df.to_string(index=False))
    print()

