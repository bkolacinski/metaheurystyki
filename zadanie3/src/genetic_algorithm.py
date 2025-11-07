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

        return initial_population

    def run(self,
            cross_probability: float,
            mutation_probability: float,
            population_size: int,
            iterations: int) -> None:
        pass
