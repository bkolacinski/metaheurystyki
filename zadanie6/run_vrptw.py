#!/usr/bin/env python3
"""
VRPTW Solver - Ant Colony Optimization (ACO) with Numba Acceleration

This script uses a proper ACO metaheuristic with pheromone-based learning
to solve Vehicle Routing Problems with Time Windows (VRPTW) using Solomon benchmarks.

ACO FEATURES:
    - Pheromone trails on edges (learned information)
    - Probabilistic solution construction by ants
    - Global best pheromone update strategy
    - Parallel ant execution with @njit(parallel=True)
    - Heuristic combining distance + time window urgency

USAGE:
    python run_vrptw.py [instance_name]

EXAMPLES:
    python run_vrptw.py              # Run all instances
    python run_vrptw.py c107.txt     # Run specific instance
    python run_vrptw.py r106.txt     # Run specific instance

DATA FILES:
    data/c107.txt  - Clustered time windows (100 customers)
    data/r106.txt  - Random time windows (100 customers)
    data/rc208.txt - Mixed time windows (100 customers)

OPTIMIZATION GOAL:
    PRIMARY: Minimize number of vehicles (exponential penalty)
    SECONDARY: Minimize total distance

NEW FEATURES:
    - Demand-aware heuristic (prefers high-demand customers)
    - Capacity-aware selection (fills vehicles efficiently)
    - Exponential vehicle penalty: cost = vehicles² × 100000 + distance
    - Enhanced pheromone rewards for fewer-vehicle solutions

PARAMETERS (optimized for vehicle minimization):
    - C-type: 60 ants, 400 iterations, β=3.5, ρ=0.15
    - R-type: 80 ants, 500 iterations, β=4.0, ρ=0.20
    - RC-type: 70 ants, 450 iterations, β=3.8, ρ=0.15
"""

import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.aco import aco_vrptw
from src.data_loader import read_to_solomon_data
from src.solver import calculate_total_distance, three_opt_tw
from src.utils import calculate_distance_matrix


def solve_instance(
    instance_name: str, use_local_search: bool = True, verbose: bool = True
):
    """
    Solve a single VRPTW instance.

    Args:
        instance_name: Name of the data file (e.g., "c107.txt")
        use_local_search: Apply 3-opt optimization after route construction
        verbose: Print progress information

    Returns:
        routes: List of routes (each route is list of customer indices)
        n_vehicles: Number of vehicles used
        distance: Total distance traveled
    """
    # Load data
    data = read_to_solomon_data(f"data/{instance_name}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Instance: {data['name']}")
        print(f"Customers: {len(data['coords']) - 1}")
        print(f"Capacity: {data['capacity']}")
        print(f"Available vehicles: {data['n_vehicles']}")
        print("=" * 60)

    # Calculate distance matrix
    dist_mtx = calculate_distance_matrix(data["coords"])

    # Determine ACO parameters based on instance type
    n_ants = 750
    n_iterations = 250
    beta = 1.0
    rho = 0.5

    # n_ants = 750
    # n_iterations = 250
    # beta = 2.5
    # rho = 0.5
    # c107.txt        12         1608.37
    # r106.txt        20         1888.85
    # rc208.txt       4          1079.04

    if verbose:
        print(
            f"ACO Parameters: {n_ants} ants, {n_iterations} iterations, β={beta}, ρ={rho}"
        )

    # Solve VRPTW using Ant Colony Optimization
    routes, n_veh, dist = aco_vrptw(
        distance_matrix=dist_mtx,
        service_times=data["service_times"],
        window_starts=data["window_starts"],
        window_ends=data["window_ends"],
        demands=data["demands"],
        capacity=float(data["capacity"]),
        n_vehicles_available=data["n_vehicles"],
        n_ants=n_ants,
        n_iterations=n_iterations,
        alpha=1.5,  # Increased pheromone importance for stronger learning
        beta=beta,  # Heuristic importance (distance + time windows + DEMAND)
        rho=rho,  # Adaptive evaporation rate
        q0=0.85,  # Slightly more exploration to find vehicle-minimal solutions
        verbose=verbose,
        speed_factor=1.0,
    )

    if verbose:
        print(f"\nInitial result: {n_veh} vehicles, {dist:.2f} distance")

    # Apply 3-opt local search to improve routes
    if use_local_search and routes:
        if verbose:
            print("\nApplying 3-opt local search...")
        optimized = []
        for i, route in enumerate(routes):
            if verbose:
                print(f"  Optimizing route {i+1}/{len(routes)}...")
            optimized_route = three_opt_tw(
                route.tolist(),
                dist_mtx,
                data["service_times"],
                data["window_starts"],
                data["window_ends"],
                speed_factor=1.0,
            )
            optimized.append(optimized_route)

        # Calculate new distance
        new_dist = sum(
            calculate_total_distance(r, dist_mtx) for r in optimized
        )
        if verbose:
            print(
                f"After optimization: {n_veh} vehicles, {new_dist:.2f} distance"
            )

        return optimized, n_veh, new_dist

    return routes, n_veh, dist


def main():
    """Main entry point."""
    # Determine which instances to run
    data_dir = "data"
    available_instances = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    )

    if len(sys.argv) > 1:
        # Specific instance(s) requested
        instances = [sys.argv[1]]
    else:
        # Run all instances
        instances = available_instances

    print("VRPTW Solver - Solomon Benchmark Test")
    print("=" * 60)
    print(f"Available instances: {', '.join(available_instances)}")

    results = {}
    for instance in instances:
        try:
            routes, n_veh, dist = solve_instance(instance)
            results[instance] = (n_veh, dist)
        except Exception as e:
            print(f"\nERROR with {instance}: {e}")
            import traceback

            traceback.print_exc()
            results[instance] = (None, None)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Instance':<15} {'Vehicles':<10} {'Distance':<15}")
    print("-" * 60)
    for name, (n_veh, dist) in results.items():
        if n_veh is not None:
            print(f"{name:<15} {n_veh:<10} {dist:<15.2f}")
        else:
            print(f"{name:<15} FAILED")


if __name__ == "__main__":
    main()
