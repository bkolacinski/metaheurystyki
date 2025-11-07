import pandas as pd

from genetic_algorithm import GeneticAlgorithm


def main():
    data_csv = pd.read_csv('../data/problem plecakowy dane CSV tabulatory.csv',
                           delimiter='\t', index_col=0).rename_axis(None)

    print(data_csv.head())

    ga = GeneticAlgorithm(
        selection_strategy=None,
        cross_strategy=None,
        mutation_strategy=None,
        population=None
    )


if __name__ == "__main__":
    main()
