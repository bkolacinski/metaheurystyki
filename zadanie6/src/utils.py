import numpy as np


def calculate_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
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
