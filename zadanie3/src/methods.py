from random import choices, randint, random, sample


class TournamentSelection:
    def __init__(self, tournament_size: int):
        self.tournament_size = tournament_size

    def select(
        self, population: tuple[list[int]], fitness_values: tuple[float]
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
    @staticmethod
    def select(
        population: list[list[int]], fitness_values: list[float]
    ) -> list[list[int]]:
        total_fitness = sum(fitness_values)

        if total_fitness == 0:
            return choices(population, k=len(population))

        probabilities = [f / total_fitness for f in fitness_values]

        parents = choices(population, weights=probabilities, k=len(population))
        return parents


class OnePointCrossover:
    @staticmethod
    def cross(
        parents: list[list[int]], cross_probability: float
    ) -> list[list[int]]:
        children = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            if i + 1 < len(parents):
                parent2 = parents[i + 1]

                if random() < cross_probability:
                    point = randint(1, len(parent1) - 1)
                    child1 = parent1[:point] + parent2[point:]
                    child2 = parent2[:point] + parent1[point:]
                    children.extend([child1, child2])
                else:
                    children.extend([parent1, parent2])
            else:
                children.append(parent1)
        return children


class TwoPointCrossover:
    @staticmethod
    def cross(
        parents: list[list[int]], cross_probability: float
    ) -> list[list[int]]:
        children = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            if i + 1 < len(parents):
                parent2 = parents[i + 1]

                if random() < cross_probability:
                    point1 = randint(1, len(parent1) - 2)
                    point2 = randint(point1, len(parent1) - 1)
                    child1 = (
                        parent1[:point1] +
                        parent2[point1:point2] +
                        parent1[point2:]
                    )
                    child2 = (
                        parent2[:point1] +
                        parent1[point1:point2] +
                        parent2[point2:]
                    )
                    children.extend([child1, child2])
                else:
                    children.extend([parent1, parent2])
            else:
                children.append(parent1)
        return children


class BitFlipMutation:
    @staticmethod
    def mutate(
        children: list[list[int]], mutation_probability: float
    ) -> list[list[int]]:
        mutated_children = []
        for child in children:
            mutated_child = child[:]
            if random() < mutation_probability:
                index = randint(0, len(child) - 1)
                mutated_child[index] = 1 - mutated_child[index]
            mutated_children.append(mutated_child)
        return mutated_children
