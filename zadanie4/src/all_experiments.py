import os
from itertools import product

import pandas as pd

from aco import ACO
from read_data import read_data


def run_experiments_for_file(data_filename: str, results_dir: str) -> None:
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

    # Parameters to test
    m_values = [10, 20, 50]  # Number of ants
    p_random_values = [0.0, 0.05, 0.1]  # Random selection probability
    alpha_values = [0.5, 1.0, 2.0]  # Pheromone influence
    beta_values = [2.0, 5.0, 10.0]  # Heuristic influence
    iterations_values = [100, 500]  # Number of iterations
    rho_values = [0.1, 0.3, 0.5]  # Evaporation rate

    num_runs = 5  # Number of runs per configuration

    # Create file-specific results directory
    file_results_dir = os.path.join(
        results_dir,
        data_filename.replace('.txt', '')
    )
    os.makedirs(file_results_dir, exist_ok=True)

    # For summary statistics
    all_results = []

    # Key experiments - subset of full grid for reasonable runtime
    # Full grid would be: 3*3*3*3*2*3 = 486 configurations
    # We'll use targeted experiments based on the requirements

    # Experiment 1: Vary number of ants (m)
    print("\n--- Eksperyment 1: Wpływ liczby mrówek (m) ---")
    for m in [10, 20, 50, 100]:
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

    # Experiment 2: Vary p_random
    print("\n--- Eksperyment 2: Wpływ prawdopodobieństwa losowego wyboru (p_random) ---")
    for p_random in [0.0, 0.01, 0.05, 0.1]:
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

    # Experiment 3: Vary alpha
    print("\n--- Eksperyment 3: Wpływ współczynnika alpha ---")
    for alpha in [0.5, 1.0, 2.0, 5.0]:
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

    # Experiment 4: Vary beta
    print("\n--- Eksperyment 4: Wpływ współczynnika beta ---")
    for beta in [1.0, 2.0, 5.0, 10.0]:
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

    # Experiment 5: Vary iterations
    print("\n--- Eksperyment 5: Wpływ liczby iteracji ---")
    for iterations in [100, 500, 1000]:
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

    # Experiment 6: Vary rho
    print("\n--- Eksperyment 6: Wpływ współczynnika wyparowywania (rho) ---")
    for rho in [0.1, 0.3, 0.5, 0.8]:
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

    # Save summary
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
    """Run ACO algorithm multiple times with given configuration."""
    best_costs = []
    worst_costs = []
    best_paths = []
    all_histories = []
    exec_times = []

    print(f"  Konfiguracja: m={params['m']}, alpha={params['alpha']}, "
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

        # Calculate worst cost from iteration history
        worst_cost = max(result['history']) if result['history'] else result['best_cost']
        worst_costs.append(worst_cost)

    # Statistics
    stats = pd.Series(best_costs)
    avg_exec_time = sum(exec_times) / len(exec_times)

    # Best overall run
    best_idx = best_costs.index(min(best_costs))
    best_path = best_paths[best_idx]
    best_cost = best_costs[best_idx]

    print(f"    -> Średnia: {stats.mean():.2f}, Min: {stats.min():.2f}, "
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
    """Save experiment results to CSV files."""
    # Save history for each run
    for run_id, history in enumerate(results['all_histories']):
        history_df = pd.DataFrame({
            'iteration': list(range(len(history))),
            'best_cost': history,
            'run_id': run_id + 1,
            **params,
        })
        history_file = os.path.join(results_dir, f"{filename}_run{run_id + 1}.csv")
        history_df.to_csv(history_file, index=False)


def main():
    """Main function to run all experiments."""
    print("=" * 80)
    print("ALGORYTM MRÓWKOWY (ACO) - EKSPERYMENTY")
    print("=" * 80)

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Run experiments for both data files
    data_files = ["A-n32-k5.txt", "A-n80-k10.txt"]

    for data_filename in data_files:
        run_experiments_for_file(data_filename, results_dir)

    print("\n" + "=" * 80)
    print("ZAKOŃCZONO - Wyniki zapisane w katalogu results/")
    print("=" * 80)


if __name__ == "__main__":
    main()
