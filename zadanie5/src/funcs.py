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

    prod_xy = xs * ys
    prod_xy2 = prod_xy * ys
    prod_xy3 = prod_xy2 * ys

    part_1 = 1.5 - xs + prod_xy
    part_2 = 2.25 - xs + prod_xy2
    part_3 = 2.625 - xs + prod_xy3
    return part_1 * part_1 + part_2 * part_2 + part_3 * part_3
