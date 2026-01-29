import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.aco import aco_vrptw
from src.data_loader import read_to_solomon_data
from src.utils import calculate_distance_matrix

FIXED_N_ANTS = 200
FIXED_N_ITERATIONS = 100
N_RUNS_PER_CONFIG = 3

PARAMETER_GRID = {
    "C": {
        "alpha": [0.5, 1.0, 1.5, 2.0, 2.5],
        "beta": [2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        "rho": [0.05, 0.10, 0.15, 0.20, 0.25],
        "q0": [0.75, 0.80, 0.85, 0.90, 0.95],
    },
    "R": {
        "alpha": [0.5, 1.0, 1.5, 2.0, 2.5],
        "beta": [3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
        "rho": [0.10, 0.15, 0.20, 0.25, 0.30],
        "q0": [0.70, 0.75, 0.80, 0.85, 0.90],
    },
    "RC": {
        "alpha": [0.5, 1.0, 1.5, 2.0, 2.5],
        "beta": [2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        "rho": [0.05, 0.10, 0.15, 0.20, 0.25],
        "q0": [0.75, 0.80, 0.85, 0.90, 0.95],
    },
}

BENCHMARKS = {
    "c107": 10,
    "r106": 12,
    "rc208": 3,
}


def get_instance_type(instance_name: str) -> str:
    return instance_name[0].upper()


def test_configuration_wrapper(args):
    instance_name, data, dist_matrix, config, run_number, total_runs = args
    return test_configuration(
        data, dist_matrix, config
    )


def test_configuration(
    data: Dict,
    dist_matrix,
    config: Dict,
) -> Tuple[int | float, float, float, float, float]:
    start_time = time.time()
    vehicles_results = []
    distance_results = []

    for _ in range(N_RUNS_PER_CONFIG):
        try:
            _, n_veh, dist = aco_vrptw(
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

        except Exception:
            vehicles_results.append(float("inf"))
            distance_results.append(float("inf"))

    elapsed = time.time() - start_time

    valid_vehicles = [v for v in vehicles_results if v != float("inf")]
    valid_distances = [d for d in distance_results if d != float("inf")]

    if valid_vehicles:
        best_vehicles = min(valid_vehicles)
        avg_vehicles = sum(valid_vehicles) / len(valid_vehicles)
        best_distance = min(valid_distances)
        avg_distance = sum(valid_distances) / len(valid_distances)

        return (
            best_vehicles,
            best_distance,
            elapsed,
            avg_vehicles,
            avg_distance,
        )
    else:
        return float("inf"), float("inf"), elapsed, float("inf"), float("inf")


def tune_instance(
    instance_name: str,
    max_configs: int | None = None,
    quick_mode: bool = False,
    n_workers: int | None = None,
) -> Dict:
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

    param_grid = PARAMETER_GRID.get(instance_type, PARAMETER_GRID["C"])

    if quick_mode:
        param_grid = {
            "alpha": param_grid["alpha"][::2],
            "beta": param_grid["beta"][::2],
            "rho": param_grid["rho"][::2],
            "q0": param_grid["q0"][::2],
        }
        print("QUICK MODE: Using reduced parameter grid")

    keys = param_grid.keys()
    values = param_grid.values()
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    if max_configs and len(configs) > max_configs:
        import random

        configs = random.sample(configs, max_configs)
        print(
            f"Testing {max_configs} random configurations (out of {len(configs)} total)"
        )
    else:
        print(f"Testing {len(configs)} parameter combinations")

    results = []
    best_vehicles = float("inf")
    best_config = None
    best_distance = float("inf")

    start_time = time.time()

    args_list = [
        (instance_name, data, dist_matrix, config, i, len(configs))
        for i, config in enumerate(configs, 1)
    ]

    if n_workers is None:
        n_workers = os.cpu_count()

    print(f"Using {n_workers} parallel workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_config = {
            executor.submit(test_configuration_wrapper, args): (
                args[3],
                args[4],
            )
            for args in args_list
        }

        completed = 0
        for future in as_completed(future_to_config):
            config, _ = future_to_config[future]
            completed += 1

            try:
                best_veh, best_dist, elapsed, avg_veh, avg_dist = (
                    future.result()
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

                is_new_best = False
                if best_veh < best_vehicles or (
                    best_veh == best_vehicles and best_dist < best_distance
                ):
                    best_vehicles = best_veh
                    best_distance = best_dist
                    best_config = config.copy()
                    is_new_best = True

                # Progress update
                status = "✓ NEW BEST" if is_new_best else ""
                print(
                    f"  [{completed}/{len(configs)}] "
                    f"α={config['alpha']}, β={config['beta']}, "
                    f"ρ={config['rho']}, q0={config['q0']} → "
                    f"{best_veh}p/{best_dist:.2f}d ({elapsed:.1f}s) {status}"
                )

            except Exception as e:
                print(f"  [{completed}/{len(configs)}] ERROR: {e}")
                results.append(
                    {
                        "config": config,
                        "n_vehicles_best": float("inf"),
                        "n_vehicles_avg": float("inf"),
                        "distance_best": float("inf"),
                        "distance_avg": float("inf"),
                        "time": 0.0,
                    }
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
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    instance = tuning_results["instance"].replace(".txt", "")

    csv_path = os.path.join(
        output_dir, f"tuning_results_{instance}_{timestamp}.csv"
    )
    with open(csv_path, "w") as f:
        f.write(
            "n_ants,n_iterations,alpha,beta,rho,q0,"
            "vehicles_best,vehicles_avg,distance_best,distance_avg,time\n"
        )

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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )

    args = parser.parse_args()

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
    print(
        f"Workers: {args.workers if args.workers else os.cpu_count()} (CPU count: {os.cpu_count()})"
    )
    print("=" * 70)

    all_results = {}

    for instance in instances:
        try:
            tuning_results = tune_instance(
                instance,
                max_configs=args.max_configs,
                quick_mode=args.quick,
                n_workers=args.workers,
            )
            all_results[instance] = tuning_results
            save_results(tuning_results, output_dir=args.output_dir)
        except Exception as e:
            print(f"\nERROR tuning {instance}: {e}")
            import traceback

            traceback.print_exc()

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
