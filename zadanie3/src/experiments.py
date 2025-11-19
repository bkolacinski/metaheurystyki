from itertools import product

import pandas as pd

from genetic_algorithm import GeneticAlgorithm
from methods import *


def main():
    data_csv = pd.read_csv(
        "../data/problem plecakowy dane CSV tabulatory.csv",
        delimiter="\t", index_col=0
    )
    data_csv.rename_axis(None)
    data_csv.columns = ["Nazwa", "Waga", "Wartosc"]
    data_csv["Waga"] = (data_csv["Waga"]
                        .astype(str)
                        .str.replace(" ", "")
                        .astype(int))
    data_csv["Wartosc"] = (
        data_csv["Wartosc"].astype(str).str.replace(" ", "").astype(int)
    )

    selection_strategies = [
        TournamentSelection(tournament_size=3),
        RouletteWheelSelection()
    ]
    cross_strategies = [
        OnePointCrossover(),
        TwoPointCrossover()
    ]
    mutation_strategy = BitFlipMutation()

    cross_probabilities = [0.6, 0.8, 1.0]
    mutation_probabilities = [0.01, 0.05, 0.1]
    population_sizes = [50, 100, 200]
    iterations = 1000

    combinations = product(
        selection_strategies,
        cross_strategies,
        cross_probabilities,
        mutation_probabilities,
        population_sizes
    )

    for (selection_strategy, cross_strategy, cross_probability,
         mutation_probability, population_size) in combinations:
        print(f'\n{"=" * 80}\n'
              f'Strategia selekcji: {selection_strategy.__class__.__name__}\n'
              f'Strategia krzyżowania: {cross_strategy.__class__.__name__}\n'
              f'Strategia mutacji: {mutation_strategy.__class__.__name__}\n'
              f'Prawdopodobieństwo krzyżowania: {cross_probability}\n'
              f'Prawdopodobieństwo mutacji: {mutation_probability}\n'
              f'Wielkość populacji: {population_size}\n'
              f'Liczba iteracji: {iterations}\n')

        ga = GeneticAlgorithm(
            selection_strategy=selection_strategy,
            cross_strategy=cross_strategy,
            mutation_strategy=mutation_strategy,
            items=data_csv
        )

        best_result = {
            "best_fitness": 0,
            "best_individual": None
        }

        worst_result = {
            "worst_fitness": float('inf'),
            "worst_individual": None
        }

        avg_result_value = 0.0

        exec_time = 0.0

        for _ in range(5):
            result, _ = ga.run(
                cross_probability=cross_probability,
                mutation_probability=mutation_probability,
                population_size=population_size,
                iterations=iterations
            )

            if result["best_fitness"] > best_result["best_fitness"]:
                best_result["best_fitness"] = result["best_fitness"]
                best_result["best_individual"] = result["best_individual"]
            if result["best_fitness"] < worst_result["worst_fitness"]:
                worst_result["worst_fitness"] = result["best_fitness"]
                worst_result["worst_individual"] = result["best_individual"]
            avg_result_value += result["best_fitness"]
            exec_time += result["execution_time"]

        print(f'Najlepszy wynik: {best_result["best_fitness"]} - '
              f'({best_result["best_individual"]})\n'
              f'Najgorszy wynik: {worst_result["worst_fitness"]} - '
              f'({worst_result["worst_individual"]})\n'
              f'Średni wynik: {avg_result_value / 5}\n'
              f'Czas wykonania 5 uruchomień: {exec_time:.3f} sekund\n')


if __name__ == "__main__":
    main()
