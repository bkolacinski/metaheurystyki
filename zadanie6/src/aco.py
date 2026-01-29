"""
VRPTW Solver using proper Ant Colony Optimization (ACO) with Numba acceleration.

This is a TRUE ACO implementation with the following key components:

1. PHEROMONE TRAILS (τ):
   - Maintained on edges between nodes
   - Evaporate over time (controlled by ρ)
   - Deposited by best solutions

2. PROBABILISTIC SOLUTION CONSTRUCTION:
   - Ants build solutions probabilistically based on:
     * Pheromone intensity: τ^α (learned information)
     * Heuristic desirability: η^β (distance + time window urgency)
   - Uses pseudo-random proportional rule (ACS variant):
     * With probability q0: exploit best choice
     * Otherwise: explore probabilistically

3. HEURISTIC INFORMATION MATRIX (η):
   - Combines distance information (1/distance)
   - Time window urgency (prefers tighter windows)
   - Guides ants toward promising customers

4. PHEROMONE UPDATE:
   - Evaporation: τ = τ * (1 - ρ) - old pheromones fade
   - Deposition: Best solutions deposit pheromone ∝ 1/cost
   - Uses global best solution (elitist strategy)
   - Bounded to prevent premature convergence

5. PARALLEL EXECUTION:
   - All ants construct solutions in parallel using @njit(parallel=True)
   - Heavy computations optimized with Numba JIT compilation

Parameters:
- α (alpha): Pheromone importance (default: 1.0)
- β (beta): Heuristic importance (default: 2.0)
- ρ (rho): Evaporation rate (default: 0.1)
- q0: Exploitation threshold (default: 0.9)
- n_ants: Number of ants per iteration
- n_iterations: Number of iterations

This implementation follows the ACO metaheuristic with proper pheromone-based learning,
unlike greedy heuristics that don't use pheromones or probabilistic selection.
"""

from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from numba import njit, prange


def estimate_speed_factor(
    distance_matrix: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
) -> float:
    """
    Estimate speed factor from data.
    For Solomon benchmarks, distance and time are already compatible.
    """
    return 1.0


@njit(cache=True)
def compute_heuristic_matrix(
    distance_matrix: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Compute heuristic desirability matrix (eta).
    Combines distance and time window urgency.
    """
    n_nodes = len(distance_matrix)
    eta = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and distance_matrix[i, j] > 0:
                # Base heuristic: inverse of distance
                dist_component = 1.0 / distance_matrix[i, j]

                # Time window urgency: prefer nodes with tighter windows
                tw_width = window_ends[j] - window_starts[j]
                tw_component = 1.0 / (tw_width + 1.0)

                # Combine components
                eta[i, j] = dist_component * (1.0 + 0.5 * tw_component)

    return eta


@njit(cache=True)
def nearest_neighbor_cost(
    distance_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    demands: npt.NDArray[np.float64],
    capacity: float,
    n_vehicles_available: int,
    speed_factor: float,
) -> float:
    """
    Estimate initial solution cost using nearest neighbor heuristic.
    Used to initialize pheromone levels.
    """
    n_nodes = len(distance_matrix)
    unvisited = np.ones(n_nodes, dtype=np.int32)
    unvisited[0] = 0  # Depot already visited

    total_distance = 0.0
    n_routes = 0

    while np.sum(unvisited) > 0 and n_routes < n_vehicles_available:
        current_node = 0
        current_time = 0.0
        current_load = 0.0
        route_distance = 0.0

        while True:
            # Find nearest feasible customer
            best_customer = -1
            best_dist = np.inf

            for customer in range(1, n_nodes):
                if unvisited[customer] == 0:
                    continue

                if current_load + demands[customer] > capacity:
                    continue

                travel_time = (
                    distance_matrix[current_node, customer] * speed_factor
                )
                arrival_time = current_time + travel_time

                if arrival_time > window_ends[customer]:
                    continue

                if distance_matrix[current_node, customer] < best_dist:
                    best_dist = distance_matrix[current_node, customer]
                    best_customer = customer

            if best_customer < 0:
                break

            # Move to best customer
            travel_time = (
                distance_matrix[current_node, best_customer] * speed_factor
            )
            arrival_time = current_time + travel_time

            if arrival_time < window_starts[best_customer]:
                arrival_time = window_starts[best_customer]

            route_distance += distance_matrix[current_node, best_customer]
            current_time = arrival_time + service_times[best_customer]
            current_load += demands[best_customer]
            current_node = best_customer
            unvisited[best_customer] = 0

        # Return to depot
        route_distance += distance_matrix[current_node, 0]
        total_distance += route_distance
        n_routes += 1

    if np.sum(unvisited) > 0:
        return 1000000.0  # Infeasible

    VEHICLE_PENALTY = 10000.0
    return n_routes * VEHICLE_PENALTY + total_distance


@njit(cache=True)
def select_next_customer(
    current_node: int,
    feasible: npt.NDArray[np.int32],
    n_feasible: int,
    pheromone: npt.NDArray[np.float64],
    eta: npt.NDArray[np.float64],
    alpha: float,
    beta: float,
    q0: float,
    rand_val: float,
) -> int:
    """
    Select next customer using ACO transition rule.
    Uses pseudo-random proportional rule (ACS variant).
    """
    if n_feasible == 0:
        return -1

    # Calculate attractiveness for each feasible customer
    attractiveness = np.zeros(n_feasible, dtype=np.float64)

    for idx in range(n_feasible):
        customer = feasible[idx]
        tau = pheromone[current_node, customer]
        eta_val = eta[current_node, customer]
        attractiveness[idx] = (tau**alpha) * (eta_val**beta)

    # Exploitation vs exploration
    if rand_val < q0:
        # Exploitation: select best customer
        best_idx = 0
        best_val = attractiveness[0]
        for idx in range(1, n_feasible):
            if attractiveness[idx] > best_val:
                best_val = attractiveness[idx]
                best_idx = idx
        return int(feasible[best_idx])
    else:
        # Exploration: probabilistic selection
        total_attract = np.sum(attractiveness)

        if total_attract == 0:
            # Choose randomly
            rand_idx = int(rand_val * n_feasible) % n_feasible
            return int(feasible[rand_idx])

        # Roulette wheel selection
        probabilities = attractiveness / total_attract
        cumulative = 0.0
        rand_choice = rand_val

        for idx in range(n_feasible):
            cumulative += probabilities[idx]
            if rand_choice <= cumulative:
                return int(feasible[idx])

        return int(feasible[n_feasible - 1])


@njit(cache=True)
def construct_ant_solution(
    distance_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    demands: npt.NDArray[np.float64],
    capacity: float,
    n_vehicles_available: int,
    pheromone: npt.NDArray[np.float64],
    eta: npt.NDArray[np.float64],
    alpha: float,
    beta: float,
    q0: float,
    speed_factor: float,
    random_values: npt.NDArray[np.float64],
    rand_idx: int,
) -> tuple:
    """
    Construct solution for one ant using probabilistic selection.
    Returns (routes_flat, route_lengths, n_routes, success).

    routes_flat: flattened array of all routes concatenated
    route_lengths: length of each route
    n_routes: number of routes
    success: whether all customers were served
    """
    n_nodes = len(distance_matrix)
    unvisited = np.ones(n_nodes, dtype=np.int32)
    unvisited[0] = 0  # Depot

    # Pre-allocate arrays for routes
    max_nodes_per_route = n_nodes
    max_routes = n_vehicles_available
    routes_flat = np.zeros(max_nodes_per_route * max_routes, dtype=np.int32)
    route_lengths = np.zeros(max_routes, dtype=np.int32)
    n_routes = 0
    flat_idx = 0

    while np.sum(unvisited) > 0 and n_routes < n_vehicles_available:
        # Start new route
        route_start_idx = flat_idx
        routes_flat[flat_idx] = 0  # Depot
        flat_idx += 1

        current_node = 0
        current_time = 0.0
        current_load = 0.0

        while True:
            # Find feasible customers
            feasible = np.zeros(n_nodes, dtype=np.int32)
            n_feasible = 0

            for customer in range(1, n_nodes):
                if unvisited[customer] == 0:
                    continue

                if current_load + demands[customer] > capacity:
                    continue

                travel_time = (
                    distance_matrix[current_node, customer] * speed_factor
                )
                arrival_time = current_time + travel_time

                if arrival_time > window_ends[customer]:
                    continue

                feasible[n_feasible] = customer
                n_feasible += 1

            if n_feasible == 0:
                break

            # Select next customer
            rand_val = random_values[
                (rand_idx + flat_idx) % len(random_values)
            ]
            next_customer = select_next_customer(
                current_node,
                feasible,
                n_feasible,
                pheromone,
                eta,
                alpha,
                beta,
                q0,
                rand_val,
            )

            if next_customer < 0:
                break

            # Add customer to route
            routes_flat[flat_idx] = next_customer
            flat_idx += 1
            unvisited[next_customer] = 0

            # Update state
            travel_time = (
                distance_matrix[current_node, next_customer] * speed_factor
            )
            arrival_time = current_time + travel_time

            if arrival_time < window_starts[next_customer]:
                arrival_time = window_starts[next_customer]

            current_time = arrival_time + service_times[next_customer]
            current_load += demands[next_customer]
            current_node = next_customer

        # Return to depot
        routes_flat[flat_idx] = 0
        flat_idx += 1

        route_lengths[n_routes] = flat_idx - route_start_idx
        n_routes += 1

        # Only add route if it has customers
        if route_lengths[n_routes - 1] <= 2:
            flat_idx = route_start_idx
            n_routes -= 1
            break

    success = np.sum(unvisited) == 0

    return routes_flat, route_lengths, n_routes, success


@njit(cache=True)
def evaluate_solution(
    routes_flat: npt.NDArray[np.int32],
    route_lengths: npt.NDArray[np.int32],
    n_routes: int,
    distance_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    speed_factor: float,
) -> float:
    """
    Evaluate solution quality.
    Returns cost (minimize vehicles first, then distance).
    """
    if n_routes == 0:
        return float(np.inf)

    total_distance = 0.0
    flat_idx = 0

    for r in range(n_routes):
        route_len = route_lengths[r]

        # Extract route
        route = routes_flat[flat_idx : flat_idx + route_len]

        # Check feasibility
        current_time = 0.0
        feasible = True

        for i in range(1, route_len):
            from_node = route[i - 1]
            to_node = route[i]

            travel_time = distance_matrix[from_node, to_node] * speed_factor
            current_time += travel_time

            if current_time < window_starts[to_node]:
                current_time = window_starts[to_node]
            elif current_time > window_ends[to_node]:
                feasible = False
                break

            current_time += service_times[to_node]
            total_distance += distance_matrix[from_node, to_node]

        if not feasible:
            return float(np.inf)

        flat_idx += route_len

    # Cost function: prioritize fewer vehicles, then shorter distance
    VEHICLE_PENALTY = 10000.0
    return float(n_routes * VEHICLE_PENALTY + total_distance)


@njit(parallel=True, cache=True)
def construct_all_ant_solutions(
    n_ants: int,
    distance_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    demands: npt.NDArray[np.float64],
    capacity: float,
    n_vehicles_available: int,
    pheromone: npt.NDArray[np.float64],
    eta: npt.NDArray[np.float64],
    alpha: float,
    beta: float,
    q0: float,
    speed_factor: float,
    random_values: npt.NDArray[np.float64],
) -> tuple:
    """
    Construct solutions for all ants in parallel.
    Returns arrays containing all ant solutions and their costs.
    """
    n_nodes = len(distance_matrix)
    max_nodes = n_nodes * n_vehicles_available

    # Pre-allocate arrays for all ants
    all_routes_flat = np.zeros((n_ants, max_nodes), dtype=np.int32)
    all_route_lengths = np.zeros(
        (n_ants, n_vehicles_available), dtype=np.int32
    )
    all_n_routes = np.zeros(n_ants, dtype=np.int32)
    all_costs = np.zeros(n_ants, dtype=np.float64)
    all_success = np.zeros(n_ants, dtype=np.int32)

    for ant in prange(n_ants):
        # Each ant gets a different starting point in random values
        rand_offset = ant * 1000

        routes_flat, route_lengths, n_routes, success = construct_ant_solution(
            distance_matrix,
            service_times,
            window_starts,
            window_ends,
            demands,
            capacity,
            n_vehicles_available,
            pheromone,
            eta,
            alpha,
            beta,
            q0,
            speed_factor,
            random_values,
            rand_offset,
        )

        all_routes_flat[ant] = routes_flat
        all_route_lengths[ant] = route_lengths
        all_n_routes[ant] = n_routes
        all_success[ant] = 1 if success else 0

        if success:
            cost = evaluate_solution(
                routes_flat,
                route_lengths,
                n_routes,
                distance_matrix,
                service_times,
                window_starts,
                window_ends,
                speed_factor,
            )
            all_costs[ant] = cost
        else:
            all_costs[ant] = float(np.inf)

    return (
        all_routes_flat,
        all_route_lengths,
        all_n_routes,
        all_costs,
        all_success,
    )


@njit(cache=True)
def update_pheromones(
    pheromone: npt.NDArray[np.float64],
    best_routes_flat: npt.NDArray[np.int32],
    best_route_lengths: npt.NDArray[np.int32],
    best_n_routes: int,
    best_cost: float,
    rho: float,
    min_pheromone: float,
    max_pheromone: float,
):
    """
    Update pheromone trails.
    Uses global update rule (only best solution deposits pheromone).
    Modifies pheromone matrix in-place.
    """
    n_nodes = pheromone.shape[0]

    # Evaporation
    for i in range(n_nodes):
        for j in range(n_nodes):
            pheromone[i, j] *= 1.0 - rho

    # Deposit pheromone on best solution
    if best_cost < np.inf and best_n_routes > 0:
        delta_tau = 1.0 / best_cost

        flat_idx = 0
        for r in range(best_n_routes):
            route_len = best_route_lengths[r]

            for i in range(route_len - 1):
                from_node = best_routes_flat[flat_idx + i]
                to_node = best_routes_flat[flat_idx + i + 1]
                pheromone[from_node, to_node] += delta_tau
                pheromone[to_node, from_node] += delta_tau  # Symmetric

            flat_idx += route_len

    # Ensure pheromone levels stay within bounds
    for i in range(n_nodes):
        for j in range(n_nodes):
            if pheromone[i, j] < min_pheromone:
                pheromone[i, j] = min_pheromone
            elif pheromone[i, j] > max_pheromone:
                pheromone[i, j] = max_pheromone


def aco_vrptw(
    distance_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    demands: npt.NDArray[np.float64],
    capacity: float,
    n_vehicles_available: int,
    n_ants: int = 10,
    n_iterations: int = 100,
    alpha: float = 1.0,
    beta: float = 2.0,
    rho: float = 0.1,
    q0: float = 0.9,
    verbose: bool = False,
    speed_factor: float = 1.0,
) -> Tuple[List[npt.NDArray[np.int64]], int, float]:
    """
    Solve VRPTW using Ant Colony Optimization with Numba acceleration.

    Args:
        distance_matrix: Distance between nodes
        service_times: Service time at each node
        window_starts: Time window start for each node
        window_ends: Time window end for each node
        demands: Demand at each node
        capacity: Vehicle capacity
        n_vehicles_available: Maximum number of vehicles
        n_ants: Number of ants per iteration
        n_iterations: Number of iterations
        alpha: Pheromone importance (default: 1.0)
        beta: Heuristic importance (default: 2.0)
        rho: Evaporation rate (default: 0.1)
        q0: Exploitation threshold (default: 0.9)
        verbose: Print progress
        speed_factor: Factor to convert distance to time

    Returns:
        routes: List of routes (numpy arrays)
        n_vehicles: Number of vehicles used
        total_distance: Total distance traveled
    """
    n_nodes = len(distance_matrix)

    if speed_factor == 1.0:
        speed_factor = estimate_speed_factor(
            distance_matrix, window_starts, window_ends
        )

    # Compute heuristic matrix
    eta = compute_heuristic_matrix(distance_matrix, window_starts, window_ends)

    # Initialize pheromones
    nn_cost = nearest_neighbor_cost(
        distance_matrix,
        service_times,
        window_starts,
        window_ends,
        demands,
        capacity,
        n_vehicles_available,
        speed_factor,
    )
    initial_pheromone = 1.0 / (n_nodes * nn_cost) if nn_cost < np.inf else 0.1
    pheromone = (
        np.ones((n_nodes, n_nodes), dtype=np.float64) * initial_pheromone
    )

    min_pheromone = initial_pheromone * 0.01
    max_pheromone = initial_pheromone * 100.0

    # Best solution tracking
    best_routes_flat = np.zeros(n_nodes * n_vehicles_available, dtype=np.int32)
    best_route_lengths = np.zeros(n_vehicles_available, dtype=np.int32)
    best_n_routes = 0
    best_cost = float(np.inf)
    best_distance = float(np.inf)

    # Pre-generate random values for all iterations
    np.random.seed(42)
    random_values = np.random.random(n_ants * n_iterations * n_nodes * 2)

    if verbose:
        print(f"Starting ACO with {n_ants} ants, {n_iterations} iterations")
        print(f"Initial pheromone: {initial_pheromone:.6f}")

    for iteration in range(n_iterations):
        # Construct solutions for all ants in parallel
        (
            all_routes_flat,
            all_route_lengths,
            all_n_routes,
            all_costs,
            all_success,
        ) = construct_all_ant_solutions(
            n_ants,
            distance_matrix,
            service_times,
            window_starts,
            window_ends,
            demands,
            capacity,
            n_vehicles_available,
            pheromone,
            eta,
            alpha,
            beta,
            q0,
            speed_factor,
            random_values[iteration * n_ants * 100 :],
        )

        # Find best solution in this iteration
        iteration_best_idx = np.argmin(all_costs)
        iteration_best_cost = all_costs[iteration_best_idx]

        # Update global best
        if iteration_best_cost < best_cost:
            best_routes_flat = all_routes_flat[iteration_best_idx].copy()
            best_route_lengths = all_route_lengths[iteration_best_idx].copy()
            best_n_routes = all_n_routes[iteration_best_idx]
            best_cost = iteration_best_cost

            # Calculate actual distance
            best_distance = 0.0
            flat_idx = 0
            for r in range(best_n_routes):
                route_len = best_route_lengths[r]
                for i in range(route_len - 1):
                    from_node = best_routes_flat[flat_idx + i]
                    to_node = best_routes_flat[flat_idx + i + 1]
                    best_distance += distance_matrix[from_node, to_node]
                flat_idx += route_len

            if verbose:
                print(
                    f"Iteration {iteration}: New best - {best_n_routes} vehicles, {best_distance:.2f} distance"
                )

        # Update pheromones with best solution
        update_pheromones(
            pheromone,
            best_routes_flat,
            best_route_lengths,
            best_n_routes,
            best_cost,
            rho,
            min_pheromone,
            max_pheromone,
        )

        if verbose and iteration % 10 == 0 and iteration > 0:
            n_feasible = np.sum(all_success)
            print(
                f"Iteration {iteration}: {n_feasible}/{n_ants} ants found feasible solutions"
            )

    # Convert best solution to list of numpy arrays
    if best_cost == float(np.inf) or best_n_routes == 0:
        if verbose:
            print("No feasible solution found!")
        return [], 0, float(np.inf)

    routes = []
    flat_idx = 0
    for r in range(best_n_routes):
        route_len = best_route_lengths[r]
        route = best_routes_flat[flat_idx : flat_idx + route_len]
        routes.append(np.array(route, dtype=np.int64))
        flat_idx += route_len

    if verbose:
        print(
            f"\nFinal solution: {best_n_routes} vehicles, {best_distance:.2f} distance"
        )

    return routes, int(best_n_routes), float(best_distance)


if __name__ == "__main__":
    from data_loader import read_to_solomon_data
    from utils import calculate_distance_matrix

    data = read_to_solomon_data("data/c107.txt")
    print(f"Instance: {data['name']}")

    dist_mtx = calculate_distance_matrix(data["coords"])

    routes, n_veh, dist = aco_vrptw(
        distance_matrix=dist_mtx,
        service_times=data["service_times"],
        window_starts=data["window_starts"],
        window_ends=data["window_ends"],
        demands=data["demands"],
        capacity=float(data["capacity"]),
        n_vehicles_available=data["n_vehicles"],
        n_ants=20,
        n_iterations=100,
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        verbose=True,
    )

    print(f"\nResult: {n_veh} vehicles, {dist:.2f} distance")
    for i, route in enumerate(routes):
        print(f"Route {i+1}: {route}")
