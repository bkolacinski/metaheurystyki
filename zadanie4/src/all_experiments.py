import os
from itertools import product

import pandas as pd

from aco import ACO
from read_data import read_data


def run_experiments_for_file(
        data_filename: str, results_dir: str
) -> list[dict]:
    """Run experiments for a single data file."""
    print(f"\n{'=' * 80}")
    print(f"EKSPERYMENTY DLA PLIKU: {data_filename}")
    print(f"{'=' * 80}")

    data_file_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        data_filename,
    )

    data, distances_matrix, index_map = read_data(data_file_path)
    print(f"Wczytano {len(data)} atrakcji z pliku {data_filename}.")

    num_runs = 5

    file_results_dir = os.path.join(
        results_dir,
        data_filename.replace('.txt', '')
    )
    os.makedirs(file_results_dir, exist_ok=True)

    all_results = []

    print("\n--- Eksperyment 1: Wpływ liczby mrówek (m) ---")
    m_values = [10, 20, 50, 100]
    for m in m_values:
        params = {
            'm': m,
            'alpha': 1.0,
            'beta': 2.0,
            'rho': 0.5,
            'iterations': 100,
            'p_random': 0.0,
        }
        results = run_configuration(
            data, distances_matrix, index_map, params, num_runs
        )
        results['experiment'] = 'm'
        results['varied_value'] = m
        all_results.append(results)
        save_experiment_results(
            file_results_dir, f"exp_m_{m}", results, params
        )

    print("\n--- Eksperyment 2: Wpływ prawdopodobieństwa losowego wyboru "
          "(p_random) ---")
    p_random_values = [0.0, 0.01, 0.05, 0.1]
    for p_random in p_random_values:
        params = {
            'm': 20,
            'alpha': 1.0,
            'beta': 2.0,
            'rho': 0.5,
            'iterations': 100,
            'p_random': p_random,
        }
        results = run_configuration(
            data, distances_matrix, index_map, params, num_runs
        )
        results['experiment'] = 'p_random'
        results['varied_value'] = p_random
        all_results.append(results)
        save_experiment_results(
            file_results_dir, f"exp_prandom_{p_random}", results, params
        )

    print("\n--- Eksperyment 3: Wpływ współczynnika alpha ---")
    alpha_values = [0.5, 1.0, 2.0, 5.0]
    for alpha in alpha_values:
        params = {
            'm': 20,
            'alpha': alpha,
            'beta': 2.0,
            'rho': 0.5,
            'iterations': 100,
            'p_random': 0.0,
        }
        results = run_configuration(
            data, distances_matrix, index_map, params, num_runs
        )
        results['experiment'] = 'alpha'
        results['varied_value'] = alpha
        all_results.append(results)
        save_experiment_results(
            file_results_dir, f"exp_alpha_{alpha}", results, params
        )

    print("\n--- Eksperyment 4: Wpływ współczynnika beta ---")
    beta_values = [1.0, 2.0, 5.0, 10.0]
    for beta in beta_values:
        params = {
            'm': 20,
            'alpha': 1.0,
            'beta': beta,
            'rho': 0.5,
            'iterations': 100,
            'p_random': 0.0,
        }
        results = run_configuration(
            data, distances_matrix, index_map, params, num_runs
        )
        results['experiment'] = 'beta'
        results['varied_value'] = beta
        all_results.append(results)
        save_experiment_results(
            file_results_dir, f"exp_beta_{beta}", results, params
        )

    print("\n--- Eksperyment 5: Wpływ liczby iteracji ---")
    iterations_values = [100, 500, 1000]
    for iterations in iterations_values:
        params = {
            'm': 20,
            'alpha': 1.0,
            'beta': 2.0,
            'rho': 0.5,
            'iterations': iterations,
            'p_random': 0.0,
        }
        results = run_configuration(
            data, distances_matrix, index_map, params, num_runs
        )
        results['experiment'] = 'iterations'
        results['varied_value'] = iterations
        all_results.append(results)
        save_experiment_results(
            file_results_dir, f"exp_iter_{iterations}", results, params
        )

    print("\n--- Eksperyment 6: Wpływ współczynnika wyparowywania (rho) ---")
    rho_values = [0.1, 0.3, 0.5, 0.8]
    for rho in rho_values:
        params = {
            'm': 20,
            'alpha': 1.0,
            'beta': 2.0,
            'rho': rho,
            'iterations': 100,
            'p_random': 0.0,
        }
        results = run_configuration(
            data, distances_matrix, index_map, params, num_runs
        )
        results['experiment'] = 'rho'
        results['varied_value'] = rho
        all_results.append(results)
        save_experiment_results(
            file_results_dir, f"exp_rho_{rho}", results, params
        )

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(
        os.path.join(file_results_dir, "summary.csv"),
        index=False
    )
    print(f"\nPodsumowanie zapisano do: {file_results_dir}/summary.csv")

    return all_results


def run_configuration(
        data, distances_matrix, index_map, params: dict, num_runs: int
) -> dict:

    best_costs = []
    worst_costs = []
    best_paths = []
    all_histories = []
    exec_times = []

    print(f"\tKonfiguracja: m={params['m']}, alpha={params['alpha']}, "
          f"beta={params['beta']}, rho={params['rho']}, "
          f"iter={params['iterations']}, p_random={params['p_random']}")

    for run_id in range(num_runs):
        aco = ACO(
            data=data,
            distances_matrix=distances_matrix,
            index_map=index_map,
            params=params,
        )
        result, exec_time = aco.run()

        best_costs.append(result['best_cost'])
        best_paths.append(result['best_path'])
        all_histories.append(result['history'])
        exec_times.append(exec_time)

        worst_cost = max(result['history']) if result['history'] else result[
            'best_cost']
        worst_costs.append(worst_cost)

    stats = pd.Series(best_costs)
    avg_exec_time = sum(exec_times) / len(exec_times)

    best_idx = best_costs.index(min(best_costs))
    best_path = best_paths[best_idx]
    best_cost = best_costs[best_idx]

    print(f"\t\tŚrednia: {stats.mean():.2f}, Min: {stats.min():.2f}, "
          f"Max: {stats.max():.2f}, Std: {stats.std():.2f}")

    return {
        'params': params,
        'mean': stats.mean(),
        'median': stats.median(),
        'min': stats.min(),
        'max': stats.max(),
        'std': stats.std(),
        'best_cost': best_cost,
        'best_path': best_path,
        'all_costs': best_costs,
        'all_histories': all_histories,
        'avg_exec_time_ms': avg_exec_time,
        'total_exec_time_ms': sum(exec_times),
    }


def save_experiment_results(
        results_dir: str, filename: str, results: dict, params: dict
) -> None:
    for run_id, history in enumerate(results['all_histories']):
        history_df = pd.DataFrame({
            'iteration': list(range(len(history))),
            'best_cost': history,
            'run_id': run_id + 1,
            **params,
        })
        history_file = os.path.join(results_dir,
                                    f"{filename}_run{run_id + 1}.csv")
        history_df.to_csv(history_file, index=False)


def main():
    print("-" * 80)
    print("ALGORYTM MRÓWKOWY (ACO) - EKSPERYMENTY")
    print("-" * 80)

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    data_files = ["A-n32-k5.txt", "A-n80-k10.txt"]

    for data_filename in data_files:
        run_experiments_for_file(data_filename, results_dir)

    print("\n" + "-" * 80)
    print("ZAKOŃCZONO - Wyniki zapisane w katalogu results/")
    print("-" * 80)


if __name__ == "__main__":
    main()
