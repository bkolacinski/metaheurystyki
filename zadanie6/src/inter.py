"""
Inter-route operators for VRPTW.

This module implements the Relocate operator which moves a customer
from one route to another while maintaining time window feasibility.
"""

import numpy as np
import numpy.typing as npt
from numba import njit, prange
from typing import List, Tuple


@njit(cache=True)
def _calculate_route_completion_time(
    route: npt.NDArray[np.int64],
    distance_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    start_idx: int = 1,
) -> tuple[float, bool]:
    """
    Calculate the completion time of a route from start_idx to end.

    Returns (completion_time, is_feasible)
    """
    current_time: float = 0.0

    for i in range(start_idx, len(route)):
        from_node = route[i - 1]
        to_node = route[i]
        travel_time = float(distance_matrix[from_node, to_node])
        current_time += travel_time

        if current_time < float(window_starts[to_node]):
            current_time = float(window_starts[to_node])
        elif current_time > float(window_ends[to_node]):
            return current_time, False

        current_time += float(service_times[to_node])

    return current_time, True


@njit(cache=True)
def _find_best_insertion_position(
    customer: int,
    target_route: npt.NDArray[np.int64],
    distance_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    start_time: float,
) -> tuple[int, float, bool]:
    """
    Find the best position to insert a customer in a route.

    Returns (best_position, additional_cost, is_feasible)
    additional_cost = extra distance/time from inserting at that position
    """
    n: int = len(target_route)

    if n == 0:
        return 0, 0.0, True

    best_position: int = 0
    best_cost: float = float('inf')
    feasible: bool = False

    for pos in range(1, n):
        from_node = target_route[pos - 1]
        to_node = target_route[pos]

        travel_to_customer = float(distance_matrix[from_node, customer])
        arrival_at_customer = start_time + travel_to_customer

        if arrival_at_customer > float(window_ends[customer]):
            continue

        if arrival_at_customer < float(window_starts[customer]):
            arrival_at_customer = float(window_starts[customer])

        travel_from_customer = float(distance_matrix[customer, to_node])
        arrival_at_next = arrival_at_customer + float(service_times[customer]) + travel_from_customer

        # Check feasibility of the rest of the route
        current_time = arrival_at_next
        route_feasible = True

        for k in range(pos + 1, n):
            next_node = target_route[k]
            travel = float(distance_matrix[target_route[k - 1], next_node])
            current_time += travel

            if current_time < float(window_starts[next_node]):
                current_time = float(window_starts[next_node])
            elif current_time > float(window_ends[next_node]):
                route_feasible = False
                break

            current_time += float(service_times[next_node])

        if not route_feasible:
            continue

        cost = (
            float(distance_matrix[from_node, customer])
            + float(distance_matrix[customer, to_node])
            - float(distance_matrix[from_node, to_node])
        )

        if cost < best_cost:
            best_cost = cost
            best_position = pos
            feasible = True

    return best_position, best_cost, feasible


@njit(cache=True, parallel=True)
def find_best_relocate_move(
    routes: list[npt.NDArray[np.int64]],
    distance_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    demands: npt.NDArray[np.float64],
    capacity: float,
) -> npt.NDArray[np.float64]:
    """
    Find the best relocate move across all routes.

    Returns [delta_cost, from_route, to_route, customer_idx, insert_position]
    Negative delta_cost means improvement.
    """
    n_routes: int = len(routes)
    best_move = np.array([0.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float64)

    for from_route_idx in prange(n_routes):
        from_route = routes[from_route_idx]

        if len(from_route) <= 2:
            continue

        for customer_pos in range(1, len(from_route) - 1):
            customer = from_route[customer_pos]

            for to_route_idx in range(n_routes):
                if from_route_idx == to_route_idx:
                    continue

                to_route = routes[to_route_idx]

                customer_demand = float(demands[customer])
                to_route_demand = 0.0
                for node in to_route:
                    to_route_demand += float(demands[node])

                if to_route_demand + customer_demand > capacity:
                    continue

                start_time: float = 0.0

                insert_pos, insert_cost, feasible = _find_best_insertion_position(
                    customer,
                    to_route,
                    distance_matrix,
                    service_times,
                    window_starts,
                    window_ends,
                    start_time,
                )

                if not feasible:
                    continue

                if customer_pos > 1:
                    prev_in_from = from_route[customer_pos - 1]
                else:
                    prev_in_from = from_route[0]

                if customer_pos < len(from_route) - 1:
                    next_in_from = from_route[customer_pos + 1]
                else:
                    next_in_from = from_route[0]

                removal_cost = (
                    float(distance_matrix[prev_in_from, next_in_from])
                    - float(distance_matrix[prev_in_from, customer])
                    - float(distance_matrix[customer, next_in_from])
                )

                total_delta = removal_cost + insert_cost

                if total_delta < best_move[0] - 1e-9:
                    best_move[0] = total_delta
                    best_move[1] = float(from_route_idx)
                    best_move[2] = float(to_route_idx)
                    best_move[3] = float(customer_pos)
                    best_move[4] = float(insert_pos)

    return best_move


@njit(cache=True)
def apply_relocate_move(
    routes: list[npt.NDArray[np.int64]],
    move: npt.NDArray[np.float64],
) -> list[npt.NDArray[np.int64]]:
    """
    Apply a relocate move to the routes.

    move = [delta_cost, from_route, to_route, customer_pos, insert_pos]
    """
    from_route_idx = int(move[1])
    to_route_idx = int(move[2])
    customer_pos = int(move[3])
    insert_pos = int(move[4])

    from_route = routes[from_route_idx].copy()
    customer = from_route[customer_pos]

    new_from_route = np.empty(len(from_route) - 1, dtype=np.int64)
    for i in range(customer_pos):
        new_from_route[i] = from_route[i]
    for i in range(customer_pos + 1, len(from_route)):
        new_from_route[i - 1] = from_route[i]

    to_route = routes[to_route_idx].copy()
    new_to_route = np.empty(len(to_route) + 1, dtype=np.int64)

    for i in range(insert_pos):
        new_to_route[i] = to_route[i]

    new_to_route[insert_pos] = customer

    for i in range(insert_pos, len(to_route)):
        new_to_route[i + 1] = to_route[i]

    routes[from_route_idx] = new_from_route
    routes[to_route_idx] = new_to_route

    return routes


def relocate_operator(
    routes: list[npt.NDArray[np.int64]],
    distance_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    demands: npt.NDArray[np.float64],
    capacity: float,
    max_iterations: int = 10,
) -> list[npt.NDArray[np.int64]]:
    """
    Iteratively apply relocate operator until no improvement.

    Returns the improved routes.
    """
    current_routes = routes.copy()

    for iteration in range(max_iterations):
        best_move = find_best_relocate_move(
            current_routes,
            distance_matrix,
            service_times,
            window_starts,
            window_ends,
            demands,
            capacity,
        )

        if best_move[0] < -1e-7:
            current_routes = apply_relocate_move(current_routes, best_move)
            print(f"Relocate iteration {iteration + 1}: improved by {-best_move[0]:.2f}")
        else:
            break

    return current_routes


@njit(cache=True)
def calculate_solution_stats(
    routes: list[npt.NDArray[np.int64]],
    distance_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
) -> tuple[int, float, int]:
    """
    Calculate statistics for a VRPTW solution.

    Returns (n_vehicles, total_distance, n_infeasible)
    """
    n_vehicles = len(routes)
    total_distance = 0.0
    n_infeasible = 0

    for route in routes:
        current_time = 0.0
        route_feasible = True

        for i in range(1, len(route)):
            from_node = route[i - 1]
            to_node = route[i]
            travel_time = float(distance_matrix[from_node, to_node])
            current_time += travel_time

            if current_time < float(window_starts[to_node]):
                current_time = float(window_starts[to_node])
            elif current_time > float(window_ends[to_node]):
                route_feasible = False
                n_infeasible += 1
                break

            current_time += float(service_times[to_node])
            total_distance += travel_time

        if route_feasible and len(route) > 1:
            total_distance += float(distance_matrix[route[-1], route[0]])

    return n_vehicles, total_distance, n_infeasible


if __name__ == "__main__":
    from data_loader import read_to_solomon_data
    from utils import calculate_distance_matrix

    data = read_to_solomon_data("../data/c107.txt")
    print(f"Instance: {data['name']}")

    dist_mtx = calculate_distance_matrix(data["coords"])

    n_customers = len(data["coords"]) - 1
    mid = n_customers // 2

    route1 = np.array([0] + list(range(1, mid + 1)) + [0], dtype=np.int64)
    route2 = np.array([0] + list(range(mid + 1, n_customers + 1)) + [0], dtype=np.int64)

    routes = [route1, route2]

    print(f"Initial routes:")
    for i, r in enumerate(routes):
        print(f"  Route {i + 1}: {r}")

    n_veh, dist, inf = calculate_solution_stats(
        routes, dist_mtx, data["service_times"],
        data["window_starts"], data["window_ends"]
    )
    print(f"\nInitial: {n_veh} vehicles, {dist:.2f} distance, {inf} infeasible")

    improved_routes = relocate_operator(
        routes,
        dist_mtx,
        data["service_times"],
        data["window_starts"],
        data["window_ends"],
        data["demands"],
        float(data["capacity"]),
        max_iterations=20,
    )

    print(f"\nImproved routes:")
    for i, r in enumerate(improved_routes):
        print(f"  Route {i + 1}: {r}")

    n_veh, dist, inf = calculate_solution_stats(
        improved_routes, dist_mtx, data["service_times"],
        data["window_starts"], data["window_ends"]
    )
    print(f"\nAfter relocate: {n_veh} vehicles, {dist:.2f} distance, {inf} infeasible")
