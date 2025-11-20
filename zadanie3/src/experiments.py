import matplotlib.pyplot as plt
import numpy as np

from genetic_algorithm import GeneticAlgorithm
from methods import *
from read_data import read_data_csv


def bar_plot(max_data: dict,
             avg_data: dict,
             title: str,
             xlabel: str,
             ylabel: str) -> None:
    labels = list(max_data.keys())
    max_values = list(max_data.values())
    avg_values = list(avg_data.values())

    x = np.arange(len(labels))

    fig, ax = plt.subplots()
    bars = ax.bar(x, max_values, 0.6, color='green')
    ax.bar(x, avg_values, 0.6, color='orange')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.close()


def run_experiment(ga_parameters: dict,
                   run_parameters: dict,
                   x: int) -> tuple[float, float]:
    best_fitness, avg_best_fitness = 0, 0

    ga = GeneticAlgorithm(
        selection_strategy=ga_parameters["selection_strategy"],
        cross_strategy=ga_parameters["cross_strategy"],
        mutation_strategy=ga_parameters["mutation_strategy"],
        items=ga_parameters["items"]
    )

    for _ in range(x):
        result, _ = ga.run(
            population_size=run_parameters["population_size"],
            cross_probability=run_parameters["cross_probability"],
            mutation_probability=run_parameters["mutation_probability"],
            iterations=run_parameters["iterations"]
        )

        if result["best_fitness"] > best_fitness:
            best_fitness = result["best_fitness"]
        avg_best_fitness += result["best_fitness"]

    return best_fitness, avg_best_fitness // x


def main():
    data_csv = read_data_csv('../data/problem plecakowy '
                             'dane CSV tabulatory.csv')

    ga_parameters = {
        "selection_strategy": TournamentSelection(tournament_size=2),
        "cross_strategy": TwoPointCrossover,
        "mutation_strategy": BitFlipMutation,
        "items": data_csv
    }

    run_parameters = {
        "population_size": 200,
        "cross_probability": 0.8,
        "mutation_probability": 0.05,
        "iterations": 1000
    }

    # TODO DOKONCZYC


if __name__ == "__main__":
    main()
