import random


class Ant:
    distances_matrix: list[list[float]] = []
    pheromones: list[list[float]] = []
    alpha: float = 1.0
    beta: float = 2.0

    @classmethod
    def configure(
        cls,
        distances_matrix: list[list[float]],
        pheromones: list[list[float]],
        alpha: float,
        beta: float,
    ):
        cls.distances_matrix = distances_matrix
        cls.pheromones = pheromones
        cls.alpha = alpha
        cls.beta = beta

    def __init__(
        self, num_attractions: int, start_node: int | None = None
    ) -> None:
        self.num_attractions = num_attractions
        if start_node is None:
            start_node = random.randint(0, num_attractions - 1)
        self.visited = [start_node]

    def get_visited(self) -> list[int]:
        return self.visited

    def get_path_length(self) -> float:
        length = 0.0
        for i in range(len(self.visited) - 1):
            l1 = self.visited[i]
            l2 = self.visited[i + 1]
            length += self.distances_matrix[l1][l2]
        return length

    @staticmethod
    def _calculate_possibilities(current_node, allowed_nodes):
        denominator = 0.0

        scores = []
        for node in allowed_nodes:
            tau = Ant.pheromones[current_node][node]
            distance = Ant.distances_matrix[current_node][node]

            if distance == 0:
                distance = 1e-10

            eta = 1.0 / distance

            score = (tau**Ant.alpha) * (eta**Ant.beta)
            scores.append(score)
            denominator += score

        if denominator == 0.0:
            return [1.0 / len(allowed_nodes)] * len(allowed_nodes)

        return [score / denominator for score in scores]

    def select_next_node(self, p_random: float) -> None:
        current_node = self.visited[-1]

        all_nodes = set(range(self.num_attractions))
        visited_nodes = set(self.visited)
        allowed_nodes = list(all_nodes - visited_nodes)

        if not allowed_nodes:
            return

        if random.random() < p_random:
            next_node = random.choice(allowed_nodes)
            self.visited.append(next_node)
            return

        possibilities = self._calculate_possibilities(
            current_node, allowed_nodes
        )

        next_node = random.choices(allowed_nodes, weights=possibilities)[0]
        self.visited.append(next_node)
