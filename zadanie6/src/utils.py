import numpy as np
import numpy.typing as npt
from numba import njit, prange

# Default speed factor for Solomon benchmarks
# In Solomon, distance and time should be in the same scale
# But the data shows time windows are ~10x larger than distances
DEFAULT_SPEED_FACTOR: float = 1.0


def timer(func):
    from time import perf_counter

    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        return result, (end_time - start_time) * 1000

    return wrapper


def calculate_distance_matrix(
    coordinates: np.ndarray,
) -> npt.NDArray[np.float64]:
    num_points: int = coordinates.shape[0]
    distance_matrix: npt.NDArray[np.float64] = np.zeros(
        (num_points, num_points), dtype=np.float64
    )

    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                distance_matrix[i, j] = np.sqrt(
                    (coordinates[i, 0] - coordinates[j, 0]) ** 2
                    + (coordinates[i, 1] - coordinates[j, 1]) ** 2
                )

    return distance_matrix


@njit(cache=True)
def calculate_route_arrival_times(
    route: npt.NDArray[np.int64],
    distance_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    speed_factor: float = 1.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], bool]:
    """
    Calculate arrival times, waiting times, and check feasibility for a route.

    Args:
        route: Customer indices in order
        distance_matrix: Precomputed distances
        service_times: Service time for each customer
        window_starts: Time window start for each customer
        window_ends: Time window end for each customer
        speed_factor: Factor to convert distance to time (distance * speed_factor = time)

    Returns:
        arrival_times: Time of arrival at each customer
        waiting_times: Waiting time before service at each customer
        feasible: True if route satisfies all time windows
    """
    n: int = len(route)
    arrival_times: npt.NDArray[np.float64] = np.zeros(n, dtype=np.float64)
    waiting_times: npt.NDArray[np.float64] = np.zeros(n, dtype=np.float64)
    feasible: bool = True

    current_time: float = 0.0

    for i in range(1, n):
        from_node: int = route[i - 1]
        to_node: int = route[i]

        # Travel time (distance * speed_factor)
        travel_time: float = float(distance_matrix[from_node, to_node]) * speed_factor
        current_time += travel_time

        # Check time window
        earliest: float = window_starts[to_node]
        latest: float = window_ends[to_node]

        if current_time < earliest:
            # Wait until service can begin
            waiting_time: float = earliest - current_time
            waiting_times[i] = waiting_time
            current_time = earliest
        elif current_time > latest:
            # Late arrival - route infeasible
            feasible = False

        arrival_times[i] = current_time

        # Service time
        current_time += float(service_times[to_node])

    return arrival_times, waiting_times, feasible


@njit(cache=True)
def calculate_route_time_info(
    route: npt.NDArray[np.int64],
    distance_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    speed_factor: float = 1.0,
) -> tuple[float, float, float, bool]:
    """
    Calculate total travel time, waiting time, completion time, and feasibility.

    Returns:
        total_travel_time: Sum of travel times
        total_waiting_time: Sum of waiting times
        completion_time: Time when route ends
        feasible: True if route satisfies all time windows
    """
    n: int = len(route)
    total_travel_time: float = 0.0
    total_waiting_time: float = 0.0
    feasible: bool = True

    current_time: float = 0.0

    for i in range(1, n):
        from_node: int = route[i - 1]
        to_node: int = route[i]

        travel_time: float = float(distance_matrix[from_node, to_node]) * speed_factor
        total_travel_time += travel_time
        current_time += travel_time

        earliest: float = window_starts[to_node]
        latest: float = window_ends[to_node]

        if current_time < earliest:
            waiting_time: float = earliest - current_time
            total_waiting_time += waiting_time
            current_time = earliest
        elif current_time > latest:
            feasible = False

        current_time += float(service_times[to_node])

    completion_time: float = current_time
    return total_travel_time, total_waiting_time, completion_time, feasible


@njit(cache=True)
def calculate_route_cost(
    route: npt.NDArray[np.int64],
    distance_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    late_penalty: float = 1000000.0,
    speed_factor: float = 1.0,
) -> float:
    """
    Calculate the cost of a route considering distance and time windows.

    Returns:
        cost: Route cost (high penalty if infeasible)
    """
    travel_time, waiting_time, completion_time, feasible = calculate_route_time_info(
        route, distance_matrix, service_times, window_starts, window_ends, speed_factor
    )

    if not feasible:
        return late_penalty

    # Cost = travel time + waiting time penalty
    return travel_time + 0.5 * waiting_time


@njit(cache=True)
def is_route_feasible(
    route: npt.NDArray[np.int64],
    distance_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    speed_factor: float = 1.0,
) -> bool:
    """Check if a route satisfies all time window constraints."""
    _, _, feasible = calculate_route_arrival_times(
        route, distance_matrix, service_times, window_starts, window_ends, speed_factor
    )
    return feasible


@njit(cache=True, parallel=True)
def calculate_all_routes_cost(
    routes: list[npt.NDArray[np.int64]],
    distance_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    speed_factor: float = 1.0,
) -> tuple[npt.NDArray[np.float64], float, int]:
    """
    Calculate cost for all routes in parallel.

    Returns:
        costs: Array of costs for each route
        total_cost: Sum of all route costs
        n_infeasible: Number of infeasible routes
    """
    n_routes: int = len(routes)
    costs: npt.NDArray[np.float64] = np.zeros(n_routes, dtype=np.float64)
    total_cost: float = 0.0
    n_infeasible: int = 0
    late_penalty = 1000000.0

    for r in prange(n_routes):
        route_cost: float = calculate_route_cost(
            routes[r], distance_matrix, service_times, window_starts, window_ends,
            late_penalty, speed_factor
        )
        costs[r] = route_cost
        total_cost += route_cost
        if route_cost >= late_penalty * 0.5:  # late_penalty threshold
            n_infeasible += 1

    return costs, total_cost, n_infeasible
