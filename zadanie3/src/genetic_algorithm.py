from random import randint


class GeneticAlgorithm:
    def __init__(self,
                 selection_strategy,
                 cross_strategy,
                 mutation_strategy,
                 population):
        self.selection_strategy = selection_strategy
        self.cross_strategy = cross_strategy
        self.mutation_strategy = mutation_strategy
        self.population = population

    def generate_initial_population(self,
                                    population_size: int):
        initial_population = []

        for i in range(population_size):
            individual = [randint(0, 1) for _ in range(26)]
            initial_population.append(individual)

        return initial_population

    def calculate_fitness(self, population) -> list[float]:
        fitness_values = []

        for individual in population:
            # TODO: calculate fitness value

        return fitness_values

    def run(self,
            cross_probability: float,
            mutation_probability: float,
            population_size: int,
            iterations: int) -> None:
        self.population = self.generate_initial_population(population_size)

        best_solution = None
        last_solutions = None

        for i in range(iterations):
            fitness_values = self.calculate_fitness(self.population)

            parents = self.selection_strategy.select(self.population,
                                                     fitness_values)

            children = self.cross_strategy.cross(parents, cross_probability)

            mutated_children = self.mutation_strategy.mutate(children,
                                                             mutation_probability)

            self.population = mutated_children
