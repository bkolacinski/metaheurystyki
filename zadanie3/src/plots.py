import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from genetic_algorithm import GeneticAlgorithm
from methods import BitFlipMutation, OnePointCrossover, TournamentSelection
from read_data import read_data_csv


def plot_1(data: list[dict],
           selection_strategy,
           cross_strategy,
           mutation_strategy,
           parameters: dict):
    result, history = GeneticAlgorithm(
        selection_strategy=selection_strategy,
        cross_strategy=cross_strategy,
        mutation_strategy=mutation_strategy,
        items=data
    ).run(
        cross_probability=parameters["cross_probability"],
        mutation_probability=parameters["mutation_probability"],
        population_size=parameters["population_size"],
        iterations=parameters["iterations"],
    )

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))

    metrics = {
        0: {
            'data': [h['best_fitness'] for h in history],
            'label': 'Max Fitness',
            'title': 'Best Fitness',
            'color': 'green'
        },
        1: {
            'data': [h['avg_fitness'] for h in history],
            'label': 'Avg Fitness',
            'title': 'Average Fitness',
            'color': 'blue'
        },
        2: {
            'data': [h['worst_fitness'] for h in history],
            'label': 'Min Fitness',
            'title': 'Worst Fitness',
            'color': 'red'
        }
    }

    iterations = [h['iteration'] for h in history]

    for idx, metric in metrics.items():
        axes[idx].plot(iterations, metric['data'],
                       label=metric['label'], color=metric['color'])
        axes[idx].set_xlabel('Iteration')
        axes[idx].set_ylabel('Fitness')
        axes[idx].set_title(metric['title'])
        axes[idx].set_yscale('log')

        y_min, y_max = min(metric['data']), max(metric['data'])
        y_range = y_max - y_min
        axes[idx].set_ylim(
            min(y_min - 0.05 * y_range, y_min),
            y_max + 0.05 * y_range
        )

        ticks = np.linspace(y_min, y_max, 5)
        axes[idx].yaxis.set_major_locator(plt.FixedLocator(ticks))
        axes[idx].yaxis.set_minor_locator(plt.NullLocator())

        axes[idx].legend()
        axes[idx].grid(True)

    plt.tight_layout()
    plt.show()


def main():
    data_csv = read_data_csv('../data/problem plecakowy'
                             ' dane CSV tabulatory.csv')

    plot_1(
        data=data_csv,
        selection_strategy=TournamentSelection(tournament_size=3),
        cross_strategy=OnePointCrossover(),
        mutation_strategy=BitFlipMutation(),
        parameters={
            "cross_probability": 0.8,
            "mutation_probability": 0.01,
            "population_size": 100,
            "iterations": 1000,
        }
    )


if __name__ == "__main__":
    main()
