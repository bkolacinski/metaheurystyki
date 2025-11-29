from random import randint


class Ant:
    visited: list[int]
    num_attractions: int
    distances_matrix: list[list[float]]

    def __init__(self, num_attractions: int,
                 distances_matrix: list[list[float]]) -> None:
        self.num_attractions = num_attractions
        self.visited = [randint(1, num_attractions)]
        Ant.distances_matrix = distances_matrix

    def get_visited(self) -> list[int]:
        return self.visited

    def get_path_length(self) -> float:
        length = 0.0
        for i in range(len(self.visited) - 1):
            length += Ant.distances_matrix[self.visited[i] - 1][
                self.visited[i + 1] - 1]
        return length

    def roulette_wheel_selection(self, pheromones) -> int:
        pass

    def visit_attraction(self, pheromones) -> None:
        pass

    def visit_random_attraction(self) -> None:
        not_visited = [x for x in range(1, self.num_attractions + 1) if
                       x not in self.visited]
        if not_visited:
            next_attraction = not_visited[randint(0, len(not_visited) - 1)]
            self.visited.append(next_attraction)
