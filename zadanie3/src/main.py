import os

from genetic_algorithm import GeneticAlgorithm
from methods import (
    BitFlipMutation,
    OnePointCrossover,
    RouletteWheelSelection,
    TournamentSelection,
    TwoPointCrossover,
)
from read_data import read_data_csv


def get_float_input(
    prompt: str, default: float = None, min_val: float = None, max_val: float = None
) -> float:
    """Pobiera liczbę zmiennoprzecinkową od użytkownika."""
    while True:
        try:
            if default is not None:
                user_input = input(f"{prompt} (domyślnie {default}): ").strip()
                value = float(user_input) if user_input else default
            else:
                value = float(input(f"{prompt}: "))

            if min_val is not None and value < min_val:
                print(f"Błąd! Wartość musi być >= {min_val}.")
                continue
            if max_val is not None and value > max_val:
                print(f"Błąd! Wartość musi być <= {max_val}.")
                continue

            return value
        except ValueError:
            print("Błąd! Podaj poprawną liczbę.")


def get_int_input(
    prompt: str, default: int = None, min_val: int = None, max_val: int = None
) -> int:
    """Pobiera liczbę całkowitą od użytkownika."""
    while True:
        try:
            if default is not None:
                user_input = input(f"{prompt} (domyślnie {default}): ").strip()
                value = int(user_input) if user_input else default
            else:
                value = int(input(f"{prompt}: "))

            if min_val is not None and value < min_val:
                print(f"Błąd! Wartość musi być >= {min_val}.")
                continue
            if max_val is not None and value > max_val:
                print(f"Błąd! Wartość musi być <= {max_val}.")
                continue

            return value
        except ValueError:
            print("Błąd! Podaj poprawną liczbę całkowitą.")


def select_selection_strategy():
    """Wybór strategii selekcji."""
    print("\n" + "=" * 50)
    print("WYBÓR STRATEGII SELEKCJI")
    print("=" * 50)
    print("1. Selekcja turniejowa (Tournament Selection)")
    print("   Wymaga podania rozmiaru turnieju")
    print()
    print("2. Selekcja ruletkowa (Roulette Wheel Selection)")
    print("   Wybór proporcjonalny do wartości funkcji przystosowania")
    print("=" * 50)

    while True:
        choice = input("Wybierz strategię selekcji (1-2): ").strip()
        match choice:
            case "1":
                tournament_size = get_int_input(
                    "Podaj rozmiar turnieju", default=3, min_val=2
                )
                return TournamentSelection(tournament_size=tournament_size)
            case "2":
                return RouletteWheelSelection()
            case _:
                print("Błąd! Wybierz opcję 1-2.")


def select_crossover_strategy():
    """Wybór strategii krzyżowania."""
    print("\n" + "=" * 50)
    print("WYBÓR STRATEGII KRZYŻOWANIA")
    print("=" * 50)
    print("1. Krzyżowanie jednopunktowe (One Point Crossover)")
    print("   Wybór jednego punktu podziału")
    print()
    print("2. Krzyżowanie dwupunktowe (Two Point Crossover)")
    print("   Wybór dwóch punktów podziału")
    print("=" * 50)

    while True:
        choice = input("Wybierz strategię krzyżowania (1-2): ").strip()
        match choice:
            case "1":
                return OnePointCrossover()
            case "2":
                return TwoPointCrossover()
            case _:
                print("Błąd! Wybierz opcję 1-2.")


def get_parameters():
    """Pobiera parametry algorytmu od użytkownika."""
    print("\n" + "=" * 50)
    print("PARAMETRY ALGORYTMU")
    print("=" * 50)

    population_size = get_int_input("Wielkość populacji", default=100, min_val=2)
    iterations = get_int_input("Liczba iteracji", default=1000, min_val=1)
    cross_probability = get_float_input(
        "Prawdopodobieństwo krzyżowania", default=0.8, min_val=0.0, max_val=1.0
    )
    mutation_probability = get_float_input(
        "Prawdopodobieństwo mutacji", default=0.05, min_val=0.0, max_val=1.0
    )

    return population_size, iterations, cross_probability, mutation_probability


def display_results(result: dict, history: list[dict], items):
    """Wyświetla wyniki działania algorytmu."""
    print("\n" + "=" * 50)
    print("WYNIKI OPTYMALIZACJI")
    print("=" * 50)

    best_individual = result["best_individual"]
    best_fitness = result["best_fitness"]
    worst_individual = result["worst_individual"]
    worst_fitness = result["worst_fitness"]
    exec_time = result["execution_time"]

    # Najlepsze rozwiązanie
    selected_items_best = [i for i, gene in enumerate(best_individual) if gene == 1]
    total_weight_best = sum(items.iloc[i]["Waga"] for i in selected_items_best)
    items_names_best = [items.iloc[i]["Nazwa"] for i in selected_items_best]

    print("\nNajlepsze rozwiązanie:")
    print(f"Wartość plecaka: {best_fitness:.0f}")
    print(f"Waga plecaka: {total_weight_best:.0f} / {GeneticAlgorithm.MAX_WEIGHT}")
    print(f"Liczba przedmiotów: {len(selected_items_best)}")
    print(f"Zawartość: {', '.join(items_names_best)}")

    # Najgorsze rozwiązanie
    if worst_individual is not None:
        selected_items_worst = [
            i for i, gene in enumerate(worst_individual) if gene == 1
        ]
        total_weight_worst = sum(items.iloc[i]["Waga"] for i in selected_items_worst)
        items_names_worst = [items.iloc[i]["Nazwa"] for i in selected_items_worst]

        print("\nNajgorsze rozwiązanie:")
        print(f"Wartość plecaka: {worst_fitness:.0f}")
        print(f"Waga plecaka: {total_weight_worst:.0f} / {GeneticAlgorithm.MAX_WEIGHT}")
        print(f"Liczba przedmiotów: {len(selected_items_worst)}")
        print(f"Zawartość: {', '.join(items_names_worst)}")

    # Statystyki z historii
    if history:
        best_fitnesses = [h["best_fitness"] for h in history]
        avg_fitnesses = [h["avg_fitness"] for h in history]

        print("\nStatystyki z przebiegu:")
        print(f"Końcowa wartość funkcji przystosowania: {best_fitness:.0f}")
        print(
            f"Średnia wartość funkcji przystosowania: {sum(avg_fitnesses) / len(avg_fitnesses):.2f}"
        )
        print(f"Najlepsza wartość w ostatniej iteracji: {best_fitnesses[-1]:.0f}")

    print(f"\nCzas wykonania: {exec_time:.3f} s")
    print("=" * 50)


def main():
    """Główna funkcja programu."""
    print("\n" + "=" * 50)
    print("ALGORYTM GENETYCZNY - PROBLEM PLECAKOWY")
    print("=" * 50)

    # Wczytanie danych
    data_file_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "problem plecakowy dane CSV tabulatory.csv",
    )

    if not os.path.exists(data_file_path):
        print(f"Błąd! Nie znaleziono pliku z danymi: {data_file_path}")
        return

    try:
        items = read_data_csv(data_file_path)
        print(f"\nWczytano {len(items)} przedmiotów z pliku.")
    except Exception as e:
        print(f"Błąd podczas wczytania danych: {e}")
        return

    while True:
        selection_strategy = select_selection_strategy()
        crossover_strategy = select_crossover_strategy()
        population_size, iterations, cross_prob, mutation_prob = get_parameters()

        # Inicjalizacja algorytmu
        ga = GeneticAlgorithm(
            selection_strategy=selection_strategy,
            cross_strategy=crossover_strategy,
            mutation_strategy=BitFlipMutation(),
            items=items,
        )

        print("\nUruchamianie algorytmu...")
        result, history = ga.run(
            cross_probability=cross_prob,
            mutation_probability=mutation_prob,
            population_size=population_size,
            iterations=iterations,
        )

        display_results(result, history, items)

        print("\nCzy chcesz uruchomić algorytm ponownie?")
        repeat = (
            input("Wpisz 'tak' aby kontynuować lub dowolny klawisz aby zakończyć: ")
            .strip()
            .lower()
        )
        if repeat not in ["tak", "t", "yes", "y"]:
            print("\nDziękujemy za skorzystanie z programu!")
            break


if __name__ == "__main__":
    main()
