import pandas as pd

from genetic_algorithm import GeneticAlgorithm
from methods import BitFlipMutation, OnePointCrossover, TournamentSelection


def main():
    data_csv = pd.read_csv(
        "../data/problem plecakowy dane CSV tabulatory.csv", delimiter="\t", index_col=0
    ).rename_axis(None)

    data_csv.columns = ["Nazwa", "Waga", "Wartosc"]

    data_csv["Waga"] = data_csv["Waga"].astype(str).str.replace(" ", "").astype(int)
    data_csv["Wartosc"] = (
        data_csv["Wartosc"].astype(str).str.replace(" ", "").astype(int)
    )

    # print(data_csv)

    ga = GeneticAlgorithm(
        selection_strategy=TournamentSelection(tournament_size=3),
        cross_strategy=OnePointCrossover(),
        mutation_strategy=BitFlipMutation(),
        items=data_csv,
    )

    result, history = ga.run(
        cross_probability=0.8,
        mutation_probability=0.1,
        population_size=50,
        iterations=1000,
    )
    print(result)


if __name__ == "__main__":
    main()
