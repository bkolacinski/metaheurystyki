import os

from aco import ACO
from read_data import read_data


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


def select_data_file():
    """Wybór pliku z danymi."""
    print("\n" + "=" * 50)
    print("WYBÓR PLIKU Z DANYMI")
    print("=" * 50)
    print("1. A-n32-k5.txt (32 atrakcje)")
    print("2. A-n80-k10.txt (80 atrakcji)")
    print("=" * 50)

    while True:
        choice = input("Wybierz plik z danymi (1-2): ").strip()
        match choice:
            case "1":
                return "A-n32-k5.txt"
            case "2":
                return "A-n80-k10.txt"
            case _:
                print("Błąd! Wybierz opcję 1-2.")


def get_parameters():
    """Pobiera parametry algorytmu od użytkownika."""
    print("\n" + "=" * 50)
    print("PARAMETRY ALGORYTMU")
    print("=" * 50)

    m = get_int_input("Liczba mrówek (m)", default=20, min_val=1)
    iterations = get_int_input("Liczba iteracji", default=100, min_val=1)
    alpha = get_float_input(
        "Wpływ feromonów (alpha)", default=1.0, min_val=0.0
    )
    beta = get_float_input(
        "Wpływ heurystyki (beta)", default=2.0, min_val=0.0
    )
    rho = get_float_input(
        "Współczynnik wyparowywania (rho)", default=0.5, min_val=0.0, max_val=1.0
    )
    p_random = get_float_input(
        "Prawdopodobieństwo losowego wyboru (p_random)",
        default=0.0, min_val=0.0, max_val=1.0
    )

    return {
        'm': m,
        'iterations': iterations,
        'alpha': alpha,
        'beta': beta,
        'rho': rho,
        'p_random': p_random,
    }


def display_results(result: dict, exec_time: float, index_map: dict):
    """Wyświetla wyniki działania algorytmu."""
    print("\n" + "=" * 50)
    print("WYNIKI OPTYMALIZACJI")
    print("=" * 50)

    best_path = result['best_path']
    best_cost = result['best_cost']

    # Convert internal indices to attraction IDs
    attraction_ids = [index_map[idx] for idx in best_path]

    print(f"\nNajlepsza znaleziona trasa:")
    print(f"Kolejność atrakcji: {' -> '.join(map(str, attraction_ids))}")
    print(f"Długość trasy: {best_cost:.2f}")
    print(f"Liczba odwiedzonych atrakcji: {len(best_path)}")
    print(f"\nCzas wykonania: {exec_time:.3f} ms")
    print("=" * 50)


def main():
    """Główna funkcja programu."""
    print("\n" + "=" * 50)
    print("ALGORYTM MRÓWKOWY (ACO) - PROBLEM KOMIWOJAŻERA")
    print("Odwiedzanie atrakcji w wesołym miasteczku")
    print("=" * 50)

    while True:
        # Wybór pliku z danymi
        data_filename = select_data_file()
        data_file_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            data_filename,
        )

        if not os.path.exists(data_file_path):
            print(f"Błąd! Nie znaleziono pliku z danymi: {data_file_path}")
            continue

        try:
            data, distances_matrix, index_map = read_data(data_file_path)
            print(f"\nWczytano {len(data)} atrakcji z pliku {data_filename}.")
        except Exception as e:
            print(f"Błąd podczas wczytania danych: {e}")
            continue

        # Pobranie parametrów algorytmu
        params = get_parameters()

        # Inicjalizacja algorytmu
        aco = ACO(
            data=data,
            distances_matrix=distances_matrix,
            index_map=index_map,
            params=params,
        )

        print("\nUruchamianie algorytmu...")
        result, exec_time = aco.run()

        display_results(result, exec_time, index_map)

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
