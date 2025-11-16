import pandas as pd
from genetic_algorithm import GeneticAlgorithm


def main():
    data_csv = pd.read_csv('../data/problem plecakowy dane CSV tabulatory.csv',
                           delimiter='\t', index_col=0).rename_axis(None)

    data_csv.columns = [
        'Nazwa',
        'Waga',
        'Wartosc'
    ]

    data_csv['Waga'] = data_csv['Waga'].astype(str).str.replace(' ', '').astype(int)
    data_csv['Wartosc'] = data_csv['Wartosc'].astype(str).str.replace(' ', '').astype(int)

    # print(data_csv)

    ga = GeneticAlgorithm(
        selection_strategy=None,
        cross_strategy=None,
        mutation_strategy=None,
        population=None,
        items=data_csv
    )

    population = ga.generate_initial_population(population_size=100)

    print(ga.calculate_fitness(population, data_csv))


if __name__ == "__main__":
    main()
