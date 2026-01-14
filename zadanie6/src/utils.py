import numpy as np
import numpy.typing as npt


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
    distance_matrix: np.ndarray = np.zeros(
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
