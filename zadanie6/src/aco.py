"""
VRPTW Solver - Greedy with local search.
Uses Solomon's insertion heuristic for route construction.
"""

import numpy as np
import numpy.typing as npt
from typing import List, Tuple
from src.utils import is_route_feasible
from src.solver import calculate_total_distance


def estimate_speed_factor(
    distance_matrix: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
) -> float:
    """
    Estimate speed factor from data.

    For Solomon benchmarks, distance and time are already compatible,
    so speed_factor should be 1.0.
    """
    # For Solomon data, use 1.0 as distances are in compatible units with time
    return 1.0


def _try_insert_customer(
    customer: int,
    route: List[int],
    current_time: float,
    distance_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    speed_factor: float,
) -> Tuple[float, float, bool]:
    """
    Try inserting a customer at the best position in the route.

    Returns (additional_time, arrival_time, is_feasible).
    additional_time = extra time from inserting at best position
    """
    n = len(route)
    if n == 0:
        return 0.0, 0.0, True

    best_additional_time = float('inf')
    best_arrival = 0.0
    best_feasible = False

    # If route is just [0], insert at position 1 (after depot, before return)
    if n == 1:
        from_node = route[0]  # depot
        travel_to = distance_matrix[from_node, customer] * speed_factor
        arrival = current_time + travel_to

        if arrival <= window_ends[customer]:
            if arrival < window_starts[customer]:
                arrival = window_starts[customer]

            # Additional time cost (just going to customer, we'll add return later)
            additional = distance_matrix[from_node, customer] * speed_factor
            return additional, arrival, True
        return float('inf'), 0.0, False

    for pos in range(1, n):
        from_node = route[pos - 1]
        to_node = route[pos]

        travel_to = distance_matrix[from_node, customer] * speed_factor
        arrival = current_time + travel_to

        if arrival > window_ends[customer]:
            continue

        if arrival < window_starts[customer]:
            arrival = window_starts[customer]

        arrival += service_times[customer]

        travel_from = distance_matrix[customer, to_node] * speed_factor
        new_arrival_next = arrival + travel_from

        # Check if rest of route remains feasible
        original_next_arrival = current_time
        for k in range(pos + 1, n):
            original_next_arrival += distance_matrix[route[k - 1], route[k]] * speed_factor
            if original_next_arrival < window_starts[route[k]]:
                original_next_arrival = window_starts[route[k]]
            original_next_arrival += service_times[route[k]]

        # Check feasibility with time shift
        check_time = new_arrival_next
        route_feasible = True
        for k in range(pos + 1, n):
            next_node = route[k]
            check_time += distance_matrix[route[k - 1], next_node] * speed_factor

            if check_time < window_starts[next_node]:
                check_time = window_starts[next_node]
            elif check_time > window_ends[next_node]:
                route_feasible = False
                break

            check_time += service_times[next_node]

        if not route_feasible:
            continue

        # Additional time cost
        additional = (
            distance_matrix[from_node, customer] * speed_factor +
            distance_matrix[customer, to_node] * speed_factor -
            distance_matrix[from_node, to_node] * speed_factor
        )

        if additional < best_additional_time:
            best_additional_time = additional
            best_arrival = arrival - service_times[customer]
            best_feasible = True

    return best_additional_time, best_arrival, best_feasible


def _get_route_time_info(
    route: List[int],
    distance_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    speed_factor: float,
) -> Tuple[float, float, float, bool]:
    """Get travel time, waiting time, completion time and feasibility of a route."""
    current_time = 0.0
    total_travel = 0.0
    total_wait = 0.0

    for i in range(1, len(route)):
        from_node = route[i - 1]
        to_node = route[i]
        travel = distance_matrix[from_node, to_node] * speed_factor
        total_travel += travel
        current_time += travel

        if current_time < window_starts[to_node]:
            total_wait += window_starts[to_node] - current_time
            current_time = window_starts[to_node]
        elif current_time > window_ends[to_node]:
            return 0.0, 0.0, 0.0, False

        current_time += service_times[to_node]

    return total_travel, total_wait, current_time, True


def build_route_insertion(
    unvisited: List[int],
    distance_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    demands: npt.NDArray[np.float64],
    capacity: float,
    speed_factor: float,
    shuffle: bool = False,
) -> Tuple[List[int], List[int]]:
    """
    Build a route using Solomon's insertion heuristic.

    Returns (route, served_customers).
    """
    import random

    route = [0]  # Start at depot
    current_load = 0.0
    current_time = 0.0
    served = []

    # Make a copy that we can modify
    candidates = unvisited.copy()
    if shuffle:
        random.shuffle(candidates)

    while candidates:
        best_customer = -1
        best_cost = float('inf')
        best_arrival = 0.0

        for customer in candidates:
            if current_load + demands[customer] > capacity:
                continue

            from_node = route[-1]
            travel_time = distance_matrix[from_node, customer] * speed_factor
            arrival_time = current_time + travel_time

            if arrival_time > window_ends[customer]:
                continue

            if arrival_time < window_starts[customer]:
                arrival_time = window_starts[customer]

            # Cost function: extra time to serve this customer
            cost = arrival_time - current_time

            if cost < best_cost:
                best_cost = cost
                best_customer = customer
                best_arrival = arrival_time

        if best_customer < 1:
            break

        # Add best customer to route
        route.append(best_customer)
        served.append(best_customer)
        current_load += demands[best_customer]

        if best_arrival < window_starts[best_customer]:
            best_arrival = window_starts[best_customer]
        current_time = best_arrival + service_times[best_customer]

        candidates.remove(best_customer)

    route.append(0)  # Return to depot
    return route, served


def build_route_insertion_full(
    unvisited: List[int],
    distance_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    demands: npt.NDArray[np.float64],
    capacity: float,
    speed_factor: float,
) -> Tuple[List[int], float]:
    """Build a route trying all insertion positions (Solomon's c1 + best position)."""
    route = [0]  # Start at depot
    current_load = 0.0

    while unvisited:
        best_customer = -1
        best_pos = -1
        best_cost = float('inf')

        for idx, customer in enumerate(unvisited):
            if current_load + demands[customer] > capacity:
                continue

            # Try all positions in the route
            for insert_pos in range(1, len(route)):
                # Simulate insertion
                new_route = route[:insert_pos] + [customer] + route[insert_pos:]

                travel, wait, end_time, feasible = _get_route_time_info(
                    new_route, distance_matrix, service_times,
                    window_starts, window_ends, speed_factor
                )

                if feasible:
                    cost = travel + 0.5 * wait  # Solomon c1 with waiting time
                    if cost < best_cost:
                        best_cost = cost
                        best_customer = customer
                        best_pos = insert_pos

        if best_customer < 0:
            break

        # Insert at best position
        for idx, c in enumerate(unvisited):
            if c == best_customer:
                unvisited.pop(idx)
                break

        route.insert(best_pos, best_customer)
        current_load += demands[best_customer]

    route.append(0)
    return route, 0.0


def solve_vrptw(
    distance_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    demands: npt.NDArray[np.float64],
    capacity: float,
    n_vehicles_available: int,
    n_iterations: int = 100,
    verbose: bool = False,
    speed_factor: float = None,
) -> Tuple[List[npt.NDArray[np.int64]], int, float]:
    """Solve VRPTW using greedy insertion heuristic with random shuffling."""
    n_nodes = len(distance_matrix)

    if speed_factor is None:
        speed_factor = estimate_speed_factor(distance_matrix, window_starts, window_ends)

    best_routes = []
    best_n_vehicles = n_vehicles_available + 1
    best_distance = float('inf')
    VEHICLE_PENALTY = 10000.0

    for iteration in range(n_iterations):
        unvisited = list(range(1, n_nodes))
        routes = []

        while unvisited and len(routes) < n_vehicles_available:
            route, served = build_route_insertion(
                unvisited,
                distance_matrix, service_times, window_starts, window_ends,
                demands, capacity, speed_factor,
                shuffle=(iteration > 0)
            )

            if len(served) == 0:
                break

            # Remove served customers from unvisited
            for customer in served:
                if customer in unvisited:
                    unvisited.remove(customer)

            routes.append(np.array(route, dtype=np.int64))

        # Check if all customers served
        if len(unvisited) > 0:
            continue

        # Evaluate
        n_veh = len(routes)
        feasible = True
        total_dist = 0.0

        for route in routes:
            feas = is_route_feasible(
                route, distance_matrix, service_times, window_starts, window_ends, speed_factor
            )
            if not feas:
                feasible = False
                break
            total_dist += calculate_total_distance(route, distance_matrix)

        if feasible:
            score = n_veh * VEHICLE_PENALTY + total_dist

            if n_veh < best_n_vehicles or (n_veh == best_n_vehicles and total_dist < best_distance):
                best_n_vehicles = n_veh
                best_distance = total_dist
                best_routes = routes
                if verbose:
                    print(f"  Iteration {iteration}: {n_veh} vehicles, {total_dist:.2f}")

    return best_routes, best_n_vehicles, best_distance


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
    verbose: bool = False,
    speed_factor: float = None,
) -> Tuple[List[npt.NDArray[np.int64]], int, float]:
    """Wrapper for VRPTW solver."""
    return solve_vrptw(
        distance_matrix=distance_matrix,
        service_times=service_times,
        window_starts=window_starts,
        window_ends=window_ends,
        demands=demands,
        capacity=capacity,
        n_vehicles_available=n_vehicles_available,
        n_iterations=n_iterations,
        verbose=verbose,
        speed_factor=speed_factor,
    )


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
        n_ants=10,
        n_iterations=100,
        verbose=True,
    )

    print(f"\nResult: {n_veh} vehicles, {dist:.2f} distance")
