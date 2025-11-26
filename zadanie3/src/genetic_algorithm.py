import time
from functools import cache, lru_cache
from random import shuffle

import numpy as np


class GeneticAlgorithm:
    MAX_WEIGHT = 6404180

    def __init__(
            self,
            selection_strategy,
            cross_strategy,
            mutation_strategy,
            items: list[dict],
    ):
        self.selection_strategy = selection_strategy
        self.cross_strategy = cross_strategy
        self.mutation_strategy = mutation_strategy
        self.items = items

        self.weights = self.items["Waga"].values
        self.values = self.items["Wartosc"].values

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
            self, population: list[list[int]]
    ) -> list[float]:
        fitness_values = []

        for individual in population:
            fitness_values.append(self.calculate_fitness_indiv(
                tuple(individual)))

        return fitness_values

    @cache
    def calculate_fitness_indiv(
            self, individual
    ) -> float:
        total_weight = sum(
            gene * weight for gene, weight in zip(individual, self.weights)
        )
        total_value = sum(gene * value for gene, value in
                          zip(individual, self.values))

        if total_weight > self.MAX_WEIGHT:
            return 0.0
        else:
            return int(total_value)

    def run(
            self,
            cross_probability: float,
            mutation_probability: float,
            population_size: int,
            iterations: int,
    ) -> tuple[dict, list[dict]]:
        start_time = time.time()
        population = self.generate_initial_population(population_size)

        history = []

        best_overall_individual = None
        best_overall_fitness = -1
        worst_overall_individual = None
        worst_overall_fitness = float('inf')

        for i in range(iterations):
            fitness_values = self.calculate_fitness(population)

            current_best_fitness = max(fitness_values)
            if current_best_fitness > best_overall_fitness:
                best_overall_fitness = current_best_fitness
                best_individual_index = fitness_values.index(
                    current_best_fitness)
                best_overall_individual = population[best_individual_index]

            valid_fitness = [(f, idx) for idx, f in enumerate(fitness_values) if f > 0]
            if valid_fitness:
                current_worst_fitness = min(valid_fitness, key=lambda x: x[0])[0]
                if current_worst_fitness < worst_overall_fitness:
                    worst_overall_fitness = current_worst_fitness
                    worst_individual_index = min(valid_fitness, key=lambda x: x[0])[1]
                    worst_overall_individual = population[worst_individual_index]

            history.append(
                {
                    "iteration": i,
                    "best_fitness": current_best_fitness,
                    "worst_fitness": min(
                        [f for f in fitness_values if f > 0],
                        default=0
                    ),
                    "avg_fitness": float(np.mean(fitness_values)),
                }
            )

            parents = self.selection_strategy.select(tuple(population),
                                                     tuple(fitness_values))
            children = self.cross_strategy.cross(parents, cross_probability)
            mutated_children = self.mutation_strategy.mutate(
                children, mutation_probability
            )

            population = mutated_children[:]

        end_time = time.time()

        final_solution = {
            "best_individual": best_overall_individual,
            "best_fitness": best_overall_fitness,
            "worst_individual": worst_overall_individual,
            "worst_fitness": worst_overall_fitness,
            "execution_time": end_time - start_time
        }
        return final_solution, history
