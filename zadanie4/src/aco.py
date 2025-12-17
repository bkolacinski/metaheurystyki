from ant import Ant
from utils import timer


class ACO:
    def __init__(self, data, distances_matrix, index_map, params):
        self.data = data
        self.distances_matrix = distances_matrix
        self.index_map = index_map
        self.num_attractions = len(data)

        self.m = params['m']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.rho = params['rho']
        self.iterations = params['iterations']
        self.p_random = params['p_random']

        self.pheromones = [
            [1.0] * self.num_attractions for _ in range(self.num_attractions)
        ]

        self.global_best_path = []
        self.global_best_cost = float('inf')
        self.history_best = []

    @timer
    def run(self) -> dict:
        for i in range(self.iterations):
            Ant.configure(
                distances_matrix=self.distances_matrix,
                pheromones=self.pheromones,
                alpha=self.alpha,
                beta=self.beta,
            )

            ants = [Ant(self.num_attractions) for _ in range(self.m)]

            for _ in range(self.num_attractions - 1):
                for ant in ants:
                    ant.select_next_node(self.p_random)

            iteration_best_cost = float('inf')

            for ant in ants:
                cost = ant.get_path_length()

                if cost < self.global_best_cost:
                    self.global_best_cost = cost
                    self.global_best_path = ant.get_visited()[:]

                if cost < iteration_best_cost:
                    iteration_best_cost = cost

            self.history_best.append(iteration_best_cost)

            self._update_pheromones(ants)

        return {
            'best_cost': self.global_best_cost,
            'best_path': self.global_best_path,
            'history': self.history_best,
        }

    def _update_pheromones(self, ants) -> None:
        evaporation_factor = 1.0 - self.rho

        for i in range(self.num_attractions):
            for j in range(self.num_attractions):
                self.pheromones[i][j] *= evaporation_factor

        for ant in ants:
            path = ant.get_visited()
            path_length = ant.get_path_length()

            if path_length > 0:
                deposit = 1.0 / path_length
                for i in range(len(path) - 1):
                    from_node = path[i]
                    to_node = path[i + 1]
                    self.pheromones[from_node][to_node] += deposit
                    self.pheromones[to_node][from_node] += deposit
