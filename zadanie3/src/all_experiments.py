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

        exec_time = 0.0

        best_fitness_values = []
        best_individuals = []
        worst_fitness_values = []
        worst_individuals = []
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

            best_fitness_values.append(result["best_fitness"])
            best_individuals.append(result["best_individual"])
            worst_fitness_values.append(result["worst_fitness"])
            worst_individuals.append(result["worst_individual"])
            exec_time += result["execution_time"]

        pd.DataFrame(all_runs_history).to_csv(filepath, index=False)

        stats = pd.Series(best_fitness_values)

        best_idx = best_fitness_values.index(max(best_fitness_values))
        best_solution = best_individuals[best_idx]
        best_value = best_fitness_values[best_idx]

        selected_items_best = [i for i, gene in enumerate(best_solution) if gene == 1]
        total_weight_best = sum(data_csv.iloc[i]["Waga"] for i in selected_items_best)
        items_names_best = [data_csv.iloc[i]["Nazwa"] for i in selected_items_best]

        worst_idx = worst_fitness_values.index(min(worst_fitness_values))
        worst_solution = worst_individuals[worst_idx]
        worst_value = worst_fitness_values[worst_idx]

        selected_items_worst = [i for i, gene in enumerate(worst_solution) if gene == 1]
        total_weight_worst = sum(data_csv.iloc[i]["Waga"] for i in selected_items_worst)
        items_names_worst = [data_csv.iloc[i]["Nazwa"] for i in selected_items_worst]

        print(f'Średnia: {stats.mean():.0f}\n'
              f'Mediana: {stats.median():.0f}\n'
              f'Min: {stats.min():.0f}\n'
              f'Max: {stats.max():.0f}\n'
              f'Odchylenie standardowe: {stats.std():.0f}\n'
              f'Czas wykonania 5 uruchomień: {exec_time:.3f}s '
              f'(średni czas na uruchomienie: {exec_time/5:.3f}s)\n'
              f'\nNajlepsze rozwiązanie:\n'
              f'Wartość plecaka: {best_value:.0f}\n'
              f'Waga plecaka: {total_weight_best:.0f} / {ga.MAX_WEIGHT}\n'
              f'Liczba przedmiotów: {len(selected_items_best)}\n'
              f'Zawartość: {", ".join(items_names_best)}\n'
              f'\nNajgorsze rozwiązanie:\n'
              f'Wartość plecaka: {worst_value:.0f}\n'
              f'Waga plecaka: {total_weight_worst:.0f} / {ga.MAX_WEIGHT}\n'
              f'Liczba przedmiotów: {len(selected_items_worst)}\n'
              f'Zawartość: {", ".join(items_names_worst)}\n')


if __name__ == "__main__":
    main()
