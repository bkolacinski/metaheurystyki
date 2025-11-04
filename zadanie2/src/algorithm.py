from collections.abc import Callable
from math import exp
from random import uniform


class SimulatedAnnealing:
    def __init__(self,
                 func: Callable[[float], float],
                 domain: tuple[float, float]):
        self.func = func
        self.domain = domain

    def run_epochs(self,
                   epochs: int,
                   attempts_per_epoch: int,
                   init_temp: float,
                   alpha: float) -> tuple[float, float]:
        left, right = self.domain
        best_x = uniform(left, right)
        best_f_x = self.func(best_x)
        temperature = init_temp

        for _ in range(epochs):
            for _ in range(attempts_per_epoch):
                x = uniform(best_x - 2 * temperature,
                            best_x + 2 * temperature)
                x = max(left, min(x, right))
                f_x = self.func(x)

                if f_x > best_f_x:
                    best_x, best_f_x = x, f_x
                else:
                    if exp((f_x - best_f_x) / temperature) > uniform(0, 1):
                        best_x, best_f_x = x, f_x

            temperature *= alpha

        return best_x, best_f_x
