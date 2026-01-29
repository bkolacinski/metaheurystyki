#!/usr/bin/env python3
"""
ACO Parameter Tuning Script for VRPTW

This script performs grid search over ACO parameters to find the best configuration
for minimizing the number of vehicles (primary objective) and distance (secondary).

USAGE:
    python tune_parameters.py [instance_name]

EXAMPLES:
    python tune_parameters.py c107.txt      # Tune for specific instance
    python tune_parameters.py               # Tune for all instances

OUTPUTS:
    - results/tuning_results_[instance]_[timestamp].csv
    - results/best_parameters_[timestamp].json
"""

import itertools
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.aco import aco_vrptw
from src.data_loader import read_to_solomon_data
from src.utils import calculate_distance_matrix

# FIXED PARAMETERS
FIXED_N_ANTS = 200
FIXED_N_ITERATIONS = 100
N_RUNS_PER_CONFIG = 1  # Run each configuration multiple times for stability

# Parameter grid for tuning (only variable parameters)
PARAMETER_GRID = {
    "C": {  # Clustered instances
        "alpha": [0.5, 1.0, 1.5, 2.0, 2.5],
        "beta": [2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        "rho": [0.05, 0.10, 0.15, 0.20, 0.25],
        "q0": [0.75, 0.80, 0.85, 0.90, 0.95],
    },
    "R": {  # Random instances
        "alpha": [0.5, 1.0, 1.5, 2.0, 2.5],
        "beta": [3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
        "rho": [0.10, 0.15, 0.20, 0.25, 0.30],
        "q0": [0.70, 0.75, 0.80, 0.85, 0.90],
    },
    "RC": {  # Mixed instances
        "alpha": [0.5, 1.0, 1.5, 2.0, 2.5],
        "beta": [2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        "rho": [0.05, 0.10, 0.15, 0.20, 0.25],
        "q0": [0.75, 0.80, 0.85, 0.90, 0.95],
    },
}

# Benchmark best known results
BENCHMARKS = {
    "c107": 10,
    "r106": 12,
    "rc208": 3,
}


def get_instance_type(instance_name: str) -> str:
    """Determine instance type (C, R, or RC)."""
    return instance_name[0].upper()


def test_configuration(
    instance_name: str,
    data: Dict,
    dist_matrix,
    config: Dict,
    run_number: int,
    total_runs: int,
) -> Tuple[int, float, float]:
    """
    Test a single parameter configuration.

    Returns:
        (n_vehicles, distance, time_elapsed)
    """
    print(
        f"  [{run_number}/{total_runs}] Testing: "
        f"α={config['alpha']}, β={config['beta']}, "
        f"ρ={config['rho']}, q0={config['q0']} "
        f"(ants={FIXED_N_ANTS}, iter={FIXED_N_ITERATIONS})"
    )

    start_time = time.time()

    # Run multiple times and collect results
    vehicles_results = []
    distance_results = []

    for run in range(N_RUNS_PER_CONFIG):
        try:
            routes, n_veh, dist = aco_vrptw(
                distance_matrix=dist_matrix,
                service_times=data["service_times"],
                window_starts=data["window_starts"],
                window_ends=data["window_ends"],
                demands=data["demands"],
                capacity=float(data["capacity"]),
                n_vehicles_available=data["n_vehicles"],
                n_ants=FIXED_N_ANTS,
                n_iterations=FIXED_N_ITERATIONS,
                alpha=config["alpha"],
                beta=config["beta"],
                rho=config["rho"],
                q0=config["q0"],
                verbose=False,
                speed_factor=1.0,
            )

            vehicles_results.append(n_veh)
            distance_results.append(dist)
            print(
                f"    Run {run+1}/{N_RUNS_PER_CONFIG}: {n_veh} vehicles, {dist:.2f} distance"
            )

        except Exception as e:
            print(f"    Run {run+1}/{N_RUNS_PER_CONFIG} ERROR: {e}")
            vehicles_results.append(float("inf"))
            distance_results.append(float("inf"))

    elapsed = time.time() - start_time

    # Calculate statistics
    valid_vehicles = [v for v in vehicles_results if v != float("inf")]
    valid_distances = [d for d in distance_results if d != float("inf")]

    if valid_vehicles:
        best_vehicles = min(valid_vehicles)
        avg_vehicles = sum(valid_vehicles) / len(valid_vehicles)
        best_distance = min(valid_distances)
        avg_distance = sum(valid_distances) / len(valid_distances)

        print(
            f"    Summary: Best={best_vehicles} veh, Avg={avg_vehicles:.1f} veh, "
            f"Best dist={best_distance:.2f}, Avg dist={avg_distance:.2f}, Time={elapsed:.1f}s"
        )
        return (
            best_vehicles,
            best_distance,
            elapsed,
            avg_vehicles,
            avg_distance,
        )
    else:
        print(f"    All runs FAILED, Time={elapsed:.1f}s")
        return float("inf"), float("inf"), elapsed, float("inf"), float("inf")


def tune_instance(
    instance_name: str, max_configs: int = None, quick_mode: bool = False
) -> Dict:
    """
    Tune parameters for a specific instance.

    Args:
        instance_name: Name of the instance file (e.g., "c107.txt")
        max_configs: Maximum number of configurations to test (None = all)
        quick_mode: If True, use smaller parameter grid for faster testing

    Returns:
        Dictionary with best configuration and all results
    """
    # Load instance data
    data = read_to_solomon_data(f"data/{instance_name}")
    dist_matrix = calculate_distance_matrix(data["coords"])

    instance_type = get_instance_type(instance_name)
    instance_key = instance_name.replace(".txt", "").lower()
    benchmark = BENCHMARKS.get(instance_key, None)

    print(f"\n{'='*70}")
    print(f"Tuning parameters for: {data['name']}")
    print(f"Instance type: {instance_type}")
    print(f"Customers: {len(data['coords']) - 1}")
    print(
        f"Benchmark: {benchmark} vehicles"
        if benchmark
        else "Benchmark: Unknown"
    )
    print(f"FIXED: n_ants={FIXED_N_ANTS}, n_iterations={FIXED_N_ITERATIONS}")
    print("=" * 70)

    # Get parameter grid for this instance type
    param_grid = PARAMETER_GRID.get(instance_type, PARAMETER_GRID["C"])

    # Quick mode: reduce grid size
    if quick_mode:
        param_grid = {
            "alpha": param_grid["alpha"][::2],  # Every 2nd value
            "beta": param_grid["beta"][::2],
            "rho": param_grid["rho"][::2],
            "q0": param_grid["q0"][::2],
        }
        print("QUICK MODE: Using reduced parameter grid")

    # Generate all parameter combinations
    keys = param_grid.keys()
    values = param_grid.values()
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Limit number of configurations if requested
    if max_configs and len(configs) > max_configs:
        import random

        random.seed(42)
        configs = random.sample(configs, max_configs)
        print(
            f"Testing {max_configs} random configurations (out of {len(configs)} total)"
        )
    else:
        print(f"Testing {len(configs)} parameter combinations")

    # Test all configurations
    results = []
    best_vehicles = float("inf")
    best_config = None
    best_distance = float("inf")

    start_time = time.time()

    for i, config in enumerate(configs, 1):
        best_veh, best_dist, elapsed, avg_veh, avg_dist = test_configuration(
            instance_name, data, dist_matrix, config, i, len(configs)
        )

        result = {
            "config": config,
            "n_vehicles_best": best_veh,
            "n_vehicles_avg": avg_veh,
            "distance_best": best_dist,
            "distance_avg": avg_dist,
            "time": elapsed,
        }
        results.append(result)

        # Update best result (prioritize vehicles, then distance)
        if best_veh < best_vehicles or (
            best_veh == best_vehicles and best_dist < best_distance
        ):
            best_vehicles = best_veh
            best_distance = best_dist
            best_config = config.copy()
            print(
                f"    *** NEW OVERALL BEST: {best_veh} vehicles, {best_dist:.2f} distance ***"
            )

    total_time = time.time() - start_time

    # Summary
    print(f"\n{'='*70}")
    print("TUNING COMPLETED")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Configurations tested: {len(configs)}")
    print(f"\nBest configuration found:")
    print(
        f"  Vehicles: {best_vehicles}"
        + (f" (benchmark: {benchmark})" if benchmark else "")
    )
    print(f"  Distance: {best_distance:.2f}")
    print(f"  Parameters:")
    for key, value in best_config.items():
        print(f"    {key}: {value}")

    return {
        "instance": instance_name,
        "instance_type": instance_type,
        "benchmark": benchmark,
        "best_config": best_config,
        "best_vehicles": (
            int(best_vehicles) if best_vehicles != float("inf") else None
        ),
        "best_distance": (
            float(best_distance) if best_distance != float("inf") else None
        ),
        "all_results": results,
        "total_time": total_time,
        "configs_tested": len(configs),
    }


def save_results(tuning_results: Dict, output_dir: str = "results"):
    """Save tuning results to files."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    instance = tuning_results["instance"].replace(".txt", "")

    # Save detailed CSV
    csv_path = os.path.join(
        output_dir, f"tuning_results_{instance}_{timestamp}.csv"
    )
    with open(csv_path, "w") as f:
        # Header
        f.write(
            "n_ants,n_iterations,alpha,beta,rho,q0,"
            "vehicles_best,vehicles_avg,distance_best,distance_avg,time\n"
        )

        # Data rows
        for result in tuning_results["all_results"]:
            config = result["config"]
            f.write(
                f"{FIXED_N_ANTS},{FIXED_N_ITERATIONS},"
                f"{config['alpha']},{config['beta']},{config['rho']},{config['q0']},"
                f"{result['n_vehicles_best']},{result['n_vehicles_avg']:.2f},"
                f"{result['distance_best']:.2f},{result['distance_avg']:.2f},"
                f"{result['time']:.2f}\n"
            )

    print(f"\nDetailed results saved to: {csv_path}")

    # Save best configuration JSON
    best_config_data = {
        "instance": tuning_results["instance"],
        "benchmark": tuning_results["benchmark"],
        "best_vehicles": tuning_results["best_vehicles"],
        "best_distance": tuning_results["best_distance"],
        "parameters": tuning_results["best_config"],
        "timestamp": timestamp,
    }

    json_path = os.path.join(
        output_dir, f"best_config_{instance}_{timestamp}.json"
    )
    with open(json_path, "w") as f:
        json.dump(best_config_data, f, indent=2)

    print(f"Best configuration saved to: {json_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Tune ACO parameters for VRPTW instances"
    )
    parser.add_argument(
        "instance",
        nargs="?",
        default=None,
        help="Instance file name (e.g., c107.txt). If not provided, tune all instances.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Maximum number of configurations to test per instance",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: test fewer parameter combinations",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results (default: results/)",
    )

    args = parser.parse_args()

    # Determine which instances to tune
    if args.instance:
        instances = [args.instance]
    else:
        data_dir = "data"
        instances = sorted(
            [f for f in os.listdir(data_dir) if f.endswith(".txt")]
        )

    print("=" * 70)
    print("ACO PARAMETER TUNING FOR VRPTW")
    print("=" * 70)
    print(f"FIXED: n_ants={FIXED_N_ANTS}, n_iterations={FIXED_N_ITERATIONS}")
    print(f"RUNS PER CONFIG: {N_RUNS_PER_CONFIG}")
    print(f"TUNING: alpha, beta, rho, q0")
    print(f"Instances to tune: {', '.join(instances)}")
    print(f"Quick mode: {args.quick}")
    print(f"Max configs per instance: {args.max_configs or 'unlimited'}")
    print("=" * 70)

    all_results = {}

    for instance in instances:
        try:
            tuning_results = tune_instance(
                instance, max_configs=args.max_configs, quick_mode=args.quick
            )
            all_results[instance] = tuning_results
            save_results(tuning_results, output_dir=args.output_dir)
        except Exception as e:
            print(f"\nERROR tuning {instance}: {e}")
            import traceback

            traceback.print_exc()

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - BEST CONFIGURATIONS")
    print("=" * 70)
    print(
        f"{'Instance':<15} {'Benchmark':<10} {'Best':<10} {'Gap':<10} {'Distance':<12}"
    )
    print("-" * 70)

    for instance, results in all_results.items():
        benchmark = results["benchmark"]
        best_veh = results["best_vehicles"]
        best_dist = results["best_distance"]

        if best_veh and benchmark:
            gap = f"+{((best_veh - benchmark) / benchmark * 100):.1f}%"
        else:
            gap = "N/A"

        bench_str = str(benchmark) if benchmark else "N/A"
        veh_str = str(best_veh) if best_veh else "FAILED"
        dist_str = (
            f"{best_dist:.2f}"
            if best_dist and best_dist != float("inf")
            else "N/A"
        )

        print(
            f"{instance:<15} {bench_str:<10} {veh_str:<10} {gap:<10} {dist_str:<12}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
