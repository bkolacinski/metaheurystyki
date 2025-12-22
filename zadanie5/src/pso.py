from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc
from utils import timer


class PSO:
    def __init__(
        self,
        swarm_size: int,
        bounds: NDArray[np.floating],
        func: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        w: float,
        c1: float,
        c2: float,
        maximize: bool = True,
        use_randomness: bool = True,
    ) -> None:
        self._w = w
        self._c1 = c1
        self._c2 = c2
        self._use_randomness = use_randomness
        self._bounds = bounds

        self._func = (lambda x: -func(x)) if not maximize else func

        self._dim = len(bounds)
        self._swarm_size = swarm_size

        lower_bounds: NDArray[np.floating] = bounds[:, 0]
        upper_bounds: NDArray[np.floating] = bounds[:, 1]

        sampler = qmc.LatinHypercube(d=self._dim)
        samples = sampler.random(n=swarm_size)

        self.positions: NDArray[np.floating] = qmc.scale(
            samples, lower_bounds, upper_bounds
        )

        self.velocities: NDArray[np.floating] = np.zeros(
            (swarm_size, self._dim)
        )

        self.p_best_positions: NDArray[np.floating] = self.positions.copy()

        self.p_best_fitness: NDArray[np.floating] = np.full(
            swarm_size, -np.inf
        )

        self.g_best_position: NDArray[np.floating] = np.zeros(self._dim)

        self.g_best_fitness: float = -float("inf")

        self._evaluate()

    def _update_velocity(self) -> None:
        if self._use_randomness:
            r1 = np.random.rand(*self.positions.shape)
            r2 = np.random.rand(*self.positions.shape)
        else:
            r1 = 1.0
            r2 = 1.0

        self.velocities = (
            self._w * self.velocities
            + self._c1 * r1 * (self.p_best_positions - self.positions)
            + self._c2 * r2 * (self.g_best_position - self.positions)
        )

    def _update_position(self) -> None:
        self.positions += self.velocities
        np.clip(
            self.positions,
            self._bounds[:, 0],
            self._bounds[:, 1],
            out=self.positions,
        )

    def _evaluate(self) -> None:
        current_fitness = self._func(self.positions)

        mask = current_fitness > self.p_best_fitness
        self.p_best_positions[mask] = self.positions[mask]
        self.p_best_fitness[mask] = current_fitness[mask]

        current_best_idx = np.argmax(self.p_best_fitness)
        if self.p_best_fitness[current_best_idx] > self.g_best_fitness:
            self.g_best_fitness = self.p_best_fitness[current_best_idx]
            self.g_best_position = self.p_best_positions[
                current_best_idx
            ].copy()

    @timer
    def run(self, iterations: int) -> tuple[NDArray[np.floating], float]:
        for _ in range(iterations):
            self._update_velocity()
            self._update_position()
            self._evaluate()

        return self.g_best_position, self.g_best_fitness
