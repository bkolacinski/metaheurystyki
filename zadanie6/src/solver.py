import numpy as np
import numpy.typing as npt
from numba import njit, prange


@njit(cache=True)
def calculate_total_distance(
    route: npt.NDArray[np.int64], distance_matrix: npt.NDArray[np.float64]
) -> float:
    dist: float = 0.0
    for i in range(len(route) - 1):
        dist += float(distance_matrix[route[i], route[i + 1]])
    return dist


def apply_move(
    route: npt.NDArray[np.int64], move: npt.NDArray[np.float64]
) -> npt.NDArray[np.int64]:
    i, j, k, m_type = int(move[1]), int(move[2]), int(move[3]), int(move[4])
    s1, s2, s3, s4 = route[:i], route[i:j], route[j:k], route[k:]

    if m_type == 1:
        return np.concatenate((s1, s2[::-1], s3, s4))
    if m_type == 2:
        return np.concatenate((s1, s2, s3[::-1], s4))
    if m_type == 3:
        return np.concatenate((s1, s3, s2, s4))
    if m_type == 4:
        return np.concatenate((s1, s3, s2[::-1], s4))
    if m_type == 5:
        return np.concatenate((s1, s3[::-1], s2, s4))
    if m_type == 6:
        return np.concatenate((s1, s2[::-1], s3[::-1], s4))
    if m_type == 7:
        return np.concatenate((s1, s3[::-1], s2[::-1], s4))
    return route


def _build_3opt_route(
    route: npt.NDArray[np.int64],
    i: int,
    j: int,
    k: int,
    m_type: int,
) -> npt.NDArray[np.int64]:
    s1 = route[:i].tolist()
    s2 = route[i:j].tolist()
    s3 = route[j:k].tolist()
    s4 = route[k:].tolist()

    if m_type == 1:
        return np.array(s1 + s2[::-1] + s3 + s4, dtype=np.int64)
    if m_type == 2:
        return np.array(s1 + s2 + s3[::-1] + s4, dtype=np.int64)
    if m_type == 3:
        return np.array(s1 + s3 + s2 + s4, dtype=np.int64)
    if m_type == 4:
        return np.array(s1 + s3 + s2[::-1] + s4, dtype=np.int64)
    if m_type == 5:
        return np.array(s1 + s3[::-1] + s2 + s4, dtype=np.int64)
    if m_type == 6:
        return np.array(s1 + s2[::-1] + s3[::-1] + s4, dtype=np.int64)
    if m_type == 7:
        return np.array(s1 + s3[::-1] + s2[::-1] + s4, dtype=np.int64)
    return route.copy()


@njit(cache=True)
def _check_route_feasibility(
    route: npt.NDArray[np.int64],
    distance_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    speed_factor: float = 1.0,
) -> bool:
    current_time: float = 0.0
    for p in range(1, len(route)):
        from_node = route[p - 1]
        to_node = route[p]
        travel_time = float(distance_matrix[from_node, to_node]) * speed_factor
        current_time += travel_time

        if current_time < float(window_starts[to_node]):
            current_time = float(window_starts[to_node])
        elif current_time > float(window_ends[to_node]):
            return False

        current_time += float(service_times[to_node])
    return True


def find_best_3opt_move_tw(
    route: npt.NDArray[np.int64],
    distance_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    speed_factor: float = 1.0,
) -> npt.NDArray[np.float64]:
    n = len(route)
    best_move = np.array([0.0, -1.0, -1.0, -1.0, -1.0, 0.0])

    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                a, b = route[i - 1], route[i]
                c, d = route[j - 1], route[j]
                e, f = route[k - 1], route[k]

                d0 = (
                    distance_matrix[a, b]
                    + distance_matrix[c, d]
                    + distance_matrix[e, f]
                )

                for move_type in range(1, 8):
                    new_route = _build_3opt_route(route, i, j, k, move_type)

                    # Calculate delta distance
                    if move_type == 1:
                        d_new = (
                            distance_matrix[a, c]
                            + distance_matrix[b, d]
                            + distance_matrix[e, f]
                        )
                    elif move_type == 2:
                        d_new = (
                            distance_matrix[a, b]
                            + distance_matrix[c, e]
                            + distance_matrix[d, f]
                        )
                    elif move_type == 3:
                        d_new = (
                            distance_matrix[a, d]
                            + distance_matrix[e, b]
                            + distance_matrix[c, f]
                        )
                    elif move_type == 4:
                        d_new = (
                            distance_matrix[a, d]
                            + distance_matrix[e, c]
                            + distance_matrix[b, f]
                        )
                    elif move_type == 5:
                        d_new = (
                            distance_matrix[a, e]
                            + distance_matrix[d, b]
                            + distance_matrix[c, f]
                        )
                    elif move_type == 6:
                        d_new = (
                            distance_matrix[a, c]
                            + distance_matrix[b, e]
                            + distance_matrix[d, f]
                        )
                    else:
                        d_new = (
                            distance_matrix[a, e]
                            + distance_matrix[d, c]
                            + distance_matrix[b, f]
                        )

                    delta = d_new - d0

                    # Check time window feasibility
                    feasible = _check_route_feasibility(
                        new_route,
                        distance_matrix,
                        service_times,
                        window_starts,
                        window_ends,
                        speed_factor,
                    )

                    current_best = best_move[5]

                    if feasible and not bool(current_best):
                        if delta < best_move[0] - 1e-9:
                            best_move[0] = delta
                            best_move[1] = float(i)
                            best_move[2] = float(j)
                            best_move[3] = float(k)
                            best_move[4] = float(move_type)
                            best_move[5] = 1.0
                    elif feasible == bool(current_best):
                        if delta < best_move[0] - 1e-9:
                            best_move[0] = delta
                            best_move[1] = float(i)
                            best_move[2] = float(j)
                            best_move[3] = float(k)
                            best_move[4] = float(move_type)
                            best_move[5] = 1.0 if feasible else 0.0

    return best_move


@njit(parallel=True)
def find_best_3opt_move(
    route: npt.NDArray[np.int64], distance_matrix: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    n = len(route)
    best_move = np.array([0.0, -1.0, -1.0, -1.0, -1.0])

    for i in prange(1, n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                a, b = route[i - 1], route[i]
                c, d = route[j - 1], route[j]
                e, f = route[k - 1], route[k]

                d0 = (
                    distance_matrix[a, b]
                    + distance_matrix[c, d]
                    + distance_matrix[e, f]
                )

                d1 = (
                    distance_matrix[a, c]
                    + distance_matrix[b, d]
                    + distance_matrix[e, f]
                )
                if d1 - d0 < best_move[0]:
                    best_move[0] = d1 - d0
                    best_move[1], best_move[2], best_move[3], best_move[4] = (
                        i,
                        j,
                        k,
                        1,
                    )

                d2 = (
                    distance_matrix[a, b]
                    + distance_matrix[c, e]
                    + distance_matrix[d, f]
                )
                if d2 - d0 < best_move[0]:
                    best_move[0] = d2 - d0
                    best_move[1], best_move[2], best_move[3], best_move[4] = (
                        i,
                        j,
                        k,
                        2,
                    )

                d3 = (
                    distance_matrix[a, d]
                    + distance_matrix[e, b]
                    + distance_matrix[c, f]
                )
                if d3 - d0 < best_move[0]:
                    best_move[0] = d3 - d0
                    best_move[1], best_move[2], best_move[3], best_move[4] = (
                        i,
                        j,
                        k,
                        3,
                    )

                d4 = (
                    distance_matrix[a, d]
                    + distance_matrix[e, c]
                    + distance_matrix[b, f]
                )
                if d4 - d0 < best_move[0]:
                    best_move[0] = d4 - d0
                    best_move[1], best_move[2], best_move[3], best_move[4] = (
                        i,
                        j,
                        k,
                        4,
                    )

                d5 = (
                    distance_matrix[a, e]
                    + distance_matrix[d, b]
                    + distance_matrix[c, f]
                )
                if d5 - d0 < best_move[0]:
                    best_move[0] = d5 - d0
                    best_move[1], best_move[2], best_move[3], best_move[4] = (
                        i,
                        j,
                        k,
                        5,
                    )

                d6 = (
                    distance_matrix[a, c]
                    + distance_matrix[b, e]
                    + distance_matrix[d, f]
                )
                if d6 - d0 < best_move[0]:
                    best_move[0] = d6 - d0
                    best_move[1], best_move[2], best_move[3], best_move[4] = (
                        i,
                        j,
                        k,
                        6,
                    )

                d7 = (
                    distance_matrix[a, e]
                    + distance_matrix[d, c]
                    + distance_matrix[b, f]
                )
                if d7 - d0 < best_move[0]:
                    best_move[0] = d7 - d0
                    best_move[1], best_move[2], best_move[3], best_move[4] = (
                        i,
                        j,
                        k,
                        7,
                    )

    return best_move


def three_opt(
    initial_route: list[int], dist_matrix: npt.NDArray[np.float64]
) -> npt.NDArray[np.int64]:
    route = np.array(initial_route, dtype=np.int64)
    while True:
        best_move = find_best_3opt_move(route, dist_matrix)
        if best_move[0] < -1e-7:
            route = apply_move(route, best_move)
        else:
            break
    return route


def three_opt_tw(
    initial_route: list[int],
    dist_matrix: npt.NDArray[np.float64],
    service_times: npt.NDArray[np.float64],
    window_starts: npt.NDArray[np.float64],
    window_ends: npt.NDArray[np.float64],
    speed_factor: float = 1.0,
) -> npt.NDArray[np.int64]:
    route = np.array(initial_route, dtype=np.int64)
    while True:
        best_move = find_best_3opt_move_tw(
            route,
            dist_matrix,
            service_times,
            window_starts,
            window_ends,
            speed_factor,
        )
        if best_move[0] < -1e-7 and best_move[5] > 0.5:
            route = apply_move(route, best_move)
        else:
            break
    return route
