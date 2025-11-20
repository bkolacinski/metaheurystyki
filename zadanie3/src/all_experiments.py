import os
from itertools import product

import pandas as pd

from genetic_algorithm import GeneticAlgorithm
from methods import *
from read_data import read_data_csv


def main():
    data_csv = read_data_csv('../data/problem plecakowy'
                             ' dane CSV tabulatory.csv')

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

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

        sel_name = "Tour" if isinstance(selection_strategy, TournamentSelection) else "Roul"
        cross_name = "1P" if isinstance(cross_strategy, OnePointCrossover) else "2P"
        filename = f"{sel_name}_{cross_name}_cp{cross_probability}_mp{mutation_probability}_pop{population_size}_it{iterations}.csv"
        filepath = os.path.join(results_dir, filename)

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

        all_runs_history = []

        for i in range(5):
            result, history = ga.run(
                cross_probability=cross_probability,
                mutation_probability=mutation_probability,
                population_size=population_size,
                iterations=iterations
            )

            # Add run metadata to history
            for entry in history:
                entry['run_id'] = i + 1
                entry['execution_time'] = result['execution_time']
            all_runs_history.extend(history)

            if result["best_fitness"] > best_result["best_fitness"]:
                best_result["best_fitness"] = result["best_fitness"]
                best_result["best_individual"] = result["best_individual"]
            if result["best_fitness"] < worst_result["worst_fitness"]:
                worst_result["worst_fitness"] = result["best_fitness"]
                worst_result["worst_individual"] = result["best_individual"]
            avg_result_value += result["best_fitness"]
            exec_time += result["execution_time"]

        # Save history to CSV
        pd.DataFrame(all_runs_history).to_csv(filepath, index=False)

        print(f'Najlepszy wynik: {best_result["best_fitness"]} - '
              f'({best_result["best_individual"]})\n'
              f'Najgorszy wynik: {worst_result["worst_fitness"]} - '
              f'({worst_result["worst_individual"]})\n'
              f'Średni wynik: {avg_result_value / 5}\n'
              f'Czas wykonania 5 uruchomień: {exec_time:.3f} sekund\n')


if __name__ == "__main__":
    main()
