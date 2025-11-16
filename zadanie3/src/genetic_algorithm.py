from random import shuffle

import pandas as pd


class GeneticAlgorithm:
    MAX_WEIGHT = 6404180

    def __init__(
        self,
        selection_strategy,
        cross_strategy,
        mutation_strategy,
        population,
        items: pd.DataFrame,
    ):
        self.selection_strategy = selection_strategy
        self.cross_strategy = cross_strategy
        self.mutation_strategy = mutation_strategy
        self.population = population
        self.items = items

    def generate_initial_population(self, population_size: int) -> list[list[int]]:
        initial_population = []
        num_items = len(self.items)
        item_indices = list(range(num_items))
        weights = self.items["Waga"].values

        for _ in range(population_size):
            individual = [0] * num_items
            current_weight = 0

            shuffle(item_indices)

            for index in item_indices:
                item_weight = weights[index]
                if current_weight + item_weight <= self.MAX_WEIGHT:
                    individual[index] = 1
                    current_weight += item_weight

            initial_population.append(individual)

        return initial_population

    def calculate_fitness(
        self, population: list[list[int]], items: pd.DataFrame
    ) -> list[float]:
        fitness_values = []

        weights = items["Waga"].values
        values = items["Wartosc"].values

        for individual in population:
            total_weight = sum(
                gene * weight for gene, weight in zip(individual, weights)
            )
            total_value = sum(gene * value for gene, value in zip(individual, values))

            if total_weight > self.MAX_WEIGHT:
                fitness_values.append(0.0)
            else:
                fitness_values.append(int(total_value))

        return fitness_values

    def run(
        self,
        cross_probability: float,
        mutation_probability: float,
        population_size: int,
        iterations: int,
    ) -> None:
        self.population = self.generate_initial_population(population_size)

        best_solution = None
        last_solutions = None

        for i in range(iterations):
            fitness_values = self.calculate_fitness(self.population, self.items)

            parents = self.selection_strategy.select(self.population, fitness_values)

            children = self.cross_strategy.cross(parents, cross_probability)

            mutated_children = self.mutation_strategy.mutate(
                children, mutation_probability
            )

            self.population = mutated_children
