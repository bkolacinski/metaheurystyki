from random import choices, sample


class TournamentSelection:
    def __init__(self, tournament_size: int):
        self.tournament_size = tournament_size

    def select(
        self, population: list[list[int]], fitness_values: list[float]
    ) -> list[list[int]]:
        parents = []
        population_with_fitness = list(zip(population, fitness_values))

        for _ in population:
            tournament_contenders = sample(
                population_with_fitness, self.tournament_size
            )
            winner = max(tournament_contenders, key=lambda item: item[1])
            parents.append(winner[0])

        return parents


class RouletteWheelSelection:
    def select(
        self, population: list[list[int]], fitness_values: list[float]
    ) -> list[list[int]]:
        parents = []
        total_fitness = sum(fitness_values)

        if total_fitness == 0:
            return choices(population, k=len(population))

        probabilities = [f / total_fitness for f in fitness_values]

        parents = choices(population, weights=probabilities, k=len(population))
        return parents
