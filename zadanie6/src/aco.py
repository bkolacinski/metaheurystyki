from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from numba import njit, prange


@njit(cache=True)
def compute_heuristic_matrix(
    distance_matrix: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    demands: npt.NDArray[np.float64],
    capacity: float,
) -> npt.NDArray[np.float64]:
    n_nodes = len(distance_matrix)
    eta = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and distance_matrix[i, j] > 0:

                # Distance component: prefer closer customers
                dist_component = 1.0 / distance_matrix[i, j]

                # Time window component: prefer tighter time windows
                tw_width = window_ends[j] - window_starts[j]
                tw_component = 1.0 / (tw_width + 1.0)

                # Demand component: prefer higher demand
                demand_component = demands[j] / capacity

                eta[i, j] = dist_component * (
                    1.0 + 0.3 * tw_component + 0.7 * demand_component
                )

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
    n_nodes = len(distance_matrix)
    unvisited = np.ones(n_nodes, dtype=np.int32)
    unvisited[0] = 0

    total_distance = 0.0
    n_routes = 0

    while np.sum(unvisited) > 0 and n_routes < n_vehicles_available:
        current_node = 0
        current_time = 0.0
        current_load = 0.0
        route_distance = 0.0

        while True:
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

        route_distance += distance_matrix[current_node, 0]
        total_distance += route_distance
        n_routes += 1

    if np.sum(unvisited) > 0:
        return 1000000.0

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
    demands: npt.NDArray[np.float64],
    current_load: float,
    capacity: float,
) -> int:
    if n_feasible == 0:
        return -1

    attractiveness = np.zeros(n_feasible, dtype=np.float64)

    for idx in range(n_feasible):
        customer = feasible[idx]
        tau = pheromone[current_node, customer]
        eta_val = eta[current_node, customer]

        remaining_capacity = capacity - current_load
        demand_ratio = demands[customer] / remaining_capacity
        capacity_bonus = 1.0 + 0.5 * min(demand_ratio, 1.0)

        attractiveness[idx] = (tau**alpha) * (eta_val**beta) * capacity_bonus

    if rand_val < q0:
        best_idx = 0
        best_val = attractiveness[0]
        for idx in range(1, n_feasible):
            if attractiveness[idx] > best_val:
                best_val = attractiveness[idx]
                best_idx = idx
        return int(feasible[best_idx])
    else:
        total_attract = np.sum(attractiveness)

        if total_attract == 0:
            rand_idx = int(rand_val * n_feasible) % n_feasible
            return int(feasible[rand_idx])

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
    n_nodes = len(distance_matrix)
    unvisited = np.ones(n_nodes, dtype=np.int32)
    unvisited[0] = 0

    max_nodes_per_route = n_nodes
    max_routes = n_vehicles_available
    routes_flat = np.zeros(max_nodes_per_route * max_routes, dtype=np.int32)
    route_lengths = np.zeros(max_routes, dtype=np.int32)
    n_routes = 0
    flat_idx = 0

    while np.sum(unvisited) > 0 and n_routes < n_vehicles_available:
        route_start_idx = flat_idx
        routes_flat[flat_idx] = 0
        flat_idx += 1

        current_node = 0
        current_time = 0.0
        current_load = 0.0

        while True:
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
                demands,
                current_load,
                capacity,
            )

            if next_customer < 0:
                break

            routes_flat[flat_idx] = next_customer
            flat_idx += 1
            unvisited[next_customer] = 0

            travel_time = (
                distance_matrix[current_node, next_customer] * speed_factor
            )
            arrival_time = current_time + travel_time

            if arrival_time < window_starts[next_customer]:
                arrival_time = window_starts[next_customer]

            current_time = arrival_time + service_times[next_customer]
            current_load += demands[next_customer]
            current_node = next_customer

        routes_flat[flat_idx] = 0
        flat_idx += 1

        route_lengths[n_routes] = flat_idx - route_start_idx
        n_routes += 1

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
    if n_routes == 0:
        return float(np.inf)

    total_distance = 0.0
    flat_idx = 0

    for r in range(n_routes):
        route_len = route_lengths[r]

        route = routes_flat[flat_idx : flat_idx + route_len]

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

    VEHICLE_PENALTY = 100000.0
    vehicle_cost = (n_routes**2) * VEHICLE_PENALTY
    return float(vehicle_cost + total_distance)


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
    n_nodes = len(distance_matrix)
    max_nodes = n_nodes * n_vehicles_available

    all_routes_flat = np.zeros((n_ants, max_nodes), dtype=np.int32)
    all_route_lengths = np.zeros(
        (n_ants, n_vehicles_available), dtype=np.int32
    )
    all_n_routes = np.zeros(n_ants, dtype=np.int32)
    all_costs = np.zeros(n_ants, dtype=np.float64)
    all_success = np.zeros(n_ants, dtype=np.int32)

    for ant in prange(n_ants):
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
    n_nodes = pheromone.shape[0]

    for i in range(n_nodes):
        for j in range(n_nodes):
            pheromone[i, j] *= 1.0 - rho

    if best_cost < np.inf and best_n_routes > 0:
        vehicle_bonus = 1.0 / (best_n_routes**2)
        delta_tau = vehicle_bonus * 10000.0

        flat_idx = 0
        for r in range(best_n_routes):
            route_len = best_route_lengths[r]

            for i in range(route_len - 1):
                from_node = best_routes_flat[flat_idx + i]
                to_node = best_routes_flat[flat_idx + i + 1]
                pheromone[from_node, to_node] += delta_tau
                pheromone[to_node, from_node] += delta_tau

            flat_idx += route_len

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
    n_nodes = len(distance_matrix)

    eta = compute_heuristic_matrix(
        distance_matrix, window_starts, window_ends, demands, capacity
    )

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

    min_pheromone = initial_pheromone * 0.001
    max_pheromone = initial_pheromone * 1000.0

    best_routes_flat = np.zeros(n_nodes * n_vehicles_available, dtype=np.int32)
    best_route_lengths = np.zeros(n_vehicles_available, dtype=np.int32)
    best_n_routes = 0
    best_cost = float(np.inf)
    best_distance = float(np.inf)

    random_values = np.random.random(n_ants * n_iterations * n_nodes * 2)

    if verbose:
        print(f"Starting ACO with {n_ants} ants, {n_iterations} iterations")
        print(f"Initial pheromone: {initial_pheromone:.6f}")

    for iteration in range(n_iterations):
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

        iteration_best_idx = np.argmin(all_costs)
        iteration_best_cost = all_costs[iteration_best_idx]

        if iteration_best_cost < best_cost:
            best_routes_flat = all_routes_flat[iteration_best_idx].copy()
            best_route_lengths = all_route_lengths[iteration_best_idx].copy()
            best_n_routes = all_n_routes[iteration_best_idx]
            best_cost = iteration_best_cost

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
