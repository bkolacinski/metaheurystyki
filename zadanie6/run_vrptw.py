#!/usr/bin/env python3
"""
VRPTW Solver - Usage Example

This script demonstrates how to use the VRPTW solver with Solomon benchmark instances.

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

BENCHMARK RESULTS (from SINTEF):
    C107: 10 vehicles (our solver achieves this!)
    R106: 12 vehicles (our solver: 17, needs improvement)
    RC208: 3 vehicles (our solver achieves this!)
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import read_to_solomon_data
from utils import calculate_distance_matrix
from aco import solve_vrptw
from solver import three_opt_tw, calculate_total_distance


def solve_instance(instance_name: str, use_local_search: bool = True, verbose: bool = True):
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
        print('='*60)

    # Calculate distance matrix
    dist_mtx = calculate_distance_matrix(data["coords"])

    # Solve VRPTW
    routes, n_veh, dist = solve_vrptw(
        distance_matrix=dist_mtx,
        service_times=data["service_times"],
        window_starts=data["window_starts"],
        window_ends=data["window_ends"],
        demands=data["demands"],
        capacity=float(data["capacity"]),
        n_vehicles_available=data["n_vehicles"],
        n_iterations=200,
        verbose=verbose,
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
                route.tolist(), dist_mtx, data["service_times"],
                data["window_starts"], data["window_ends"], speed_factor=1.0
            )
            optimized.append(optimized_route)

        # Calculate new distance
        new_dist = sum(calculate_total_distance(r, dist_mtx) for r in optimized)
        if verbose:
            print(f"After optimization: {n_veh} vehicles, {new_dist:.2f} distance")

        return optimized, n_veh, new_dist

    return routes, n_veh, dist


def main():
    """Main entry point."""
    # Determine which instances to run
    data_dir = "data"
    available_instances = sorted([f for f in os.listdir(data_dir) if f.endswith('.txt')])

    if len(sys.argv) > 1:
        # Specific instance(s) requested
        instances = [sys.argv[1]]
    else:
        # Run all instances
        instances = available_instances

    print("VRPTW Solver - Solomon Benchmark Test")
    print("="*60)
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
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Instance':<15} {'Vehicles':<10} {'Distance':<15}")
    print("-"*60)
    for name, (n_veh, dist) in results.items():
        if n_veh is not None:
            print(f"{name:<15} {n_veh:<10} {dist:<15.2f}")
        else:
            print(f"{name:<15} FAILED")


if __name__ == "__main__":
    main()
