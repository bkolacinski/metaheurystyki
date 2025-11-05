from collections.abc import Callable
from math import exp
from random import uniform

from funcs import timer


class SimulatedAnnealing:
    def __init__(self,
                 func: Callable,
                 domain: tuple[float, float] | list[tuple[float, float]]):
        self.func = func
        if isinstance(domain, tuple) and len(domain) == 2 and isinstance(
                domain[0], (int, float)):
            self.domain = [domain]
        else:
            self.domain = domain
        self.dimensions = len(self.domain)

    @timer
    def run_epochs(self,
                   epochs: int,
                   attempts_per_epoch: int,
                   init_temp: float,
                   alpha: Callable[[float], float],
                   k: float) -> tuple[list[float], float, int]:

        best_point = [uniform(self.domain[i][0], self.domain[i][1])
                      for i in range(self.dimensions)]
        best_f_value = self.func(*best_point)
        temp = init_temp
        best_iters, iters = 0, 0

        for _ in range(epochs):
            for _ in range(attempts_per_epoch):
                iters += 1
                new_point = []
                for i in range(self.dimensions):
                    left, right = self.domain[i]
                    new_coord = uniform(best_point[i] - 2 * temp,
                                        best_point[i] + 2 * temp)
                    new_coord = max(left, min(new_coord, right))
                    new_point.append(new_coord)

                f_value = self.func(*new_point)  # *[6, 7] = 6, 7

                if f_value > best_f_value:
                    best_iters = iters
                    best_point, best_f_value = new_point, f_value
                else:
                    p = uniform(0, 1)
                    if exp((f_value - best_f_value) / (k * temp)) > p:
                        best_iters = iters
                        best_point, best_f_value = new_point, f_value

            temp = alpha(temp)
            if temp < 1e-10:
                break

        return best_point, best_f_value, best_iters
