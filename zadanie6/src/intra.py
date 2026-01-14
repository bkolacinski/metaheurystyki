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


if __name__ == "__main__":
    from data_loader import read_to_solomon_data
    from utils import calculate_distance_matrix

    # Wczytaj dane z pliku
    data = read_to_solomon_data("../data/c107.txt")
    print(f"Wczytano dane: {data['name']}")
    print(
        f"Liczba klientów: {len(data['coords']) - 1}"
    )  # -1 bo depot to indeks 0

    # Oblicz macierz odległości
    dist_mtx = calculate_distance_matrix(data["coords"])

    # Utwórz początkową trasę (wszystkie punkty w kolejności)
    start_route = list(range(len(data["coords"])))
    np.random.shuffle(
        start_route[1:]
    )  # Tasuj wszystko oprócz depotu (indeks 0)

    print(f"\nPoczątkowa trasa: {start_route[:10]}...")
    initial_distance = calculate_total_distance(
        np.array(start_route, dtype=np.int64), dist_mtx
    )
    print(f"Początkowy dystans: {initial_distance:.2f}")

    # Zastosuj algorytm 3-opt
    result = three_opt(start_route, dist_mtx)
    final_distance = calculate_total_distance(result, dist_mtx)

    print(f"\nKońcowa trasa: {result}")
    print(f"Końcowy dystans: {final_distance:.2f}")
    print(
        f"Poprawa: {initial_distance - final_distance:.2f} ({(1 - final_distance/initial_distance)*100:.2f}%)"
    )
