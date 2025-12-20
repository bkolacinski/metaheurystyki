import numpy as np
from numpy.typing import NDArray


# -5 <= x, y <= 5
# Global minimum is 0 at (3, 2), (−2.805118, 3.131312),
# (−3.779310, −3.283186), and (3.584428, −1.848126)
def himmelblau_function_batch(
    args: NDArray[np.floating],
) -> NDArray[np.floating]:
    xs = args[:, 0]
    ys = args[:, 1]
    part_1 = xs * xs + ys - 11
    part_2 = xs + ys * ys - 7
    return part_1 * part_1 + part_2 * part_2


# -4.5 <= x, y <= 4.5
# Global minimum is 0 at (3, 0.5)
def beale_function_batch(
    args: NDArray[np.floating],
) -> NDArray[np.floating]:
    xs = args[:, 0]
    ys = args[:, 1]

    xs_ys = xs * ys
    xs_ys_ys = xs * ys * ys
    xs_ys_ys_ys = xs * ys * ys * ys

    part_1 = 1.5 - xs + xs_ys
    part_2 = 2.25 - xs + xs_ys_ys
    part_3 = 2.625 - xs + xs_ys_ys_ys
    return part_1 * part_1 + part_2 * part_2 + part_3 * part_3
