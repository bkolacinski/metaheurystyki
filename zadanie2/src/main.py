import math
from math import sin, cos, tan, exp, log, sqrt, pi

from algorithm import SimulatedAnnealing
from funcs import func_section_3, func_section_4


def get_float_input(prompt: str, default: float = None) -> float:
    """Pobiera liczbę zmiennoprzecinkową od użytkownika."""
    while True:
        try:
            if default is not None:
                user_input = input(f"{prompt} (domyślnie {default}): ").strip()
                return float(user_input) if user_input else default
            else:
                return float(input(f"{prompt}: "))
        except ValueError:
            print("Błąd! Podaj poprawną liczbę.")


def get_int_input(prompt: str, default: int = None) -> int:
    """Pobiera liczbę całkowitą od użytkownika."""
    while True:
        try:
            if default is not None:
                user_input = input(f"{prompt} (domyślnie {default}): ").strip()
                return int(user_input) if user_input else default
            else:
                return int(input(f"{prompt}: "))
        except ValueError:
            print("Błąd! Podaj poprawną liczbę całkowitą.")


def select_function():
    """Wybór funkcji do optymalizacji."""
    print("\n" + "=" * 50)
    print("WYBÓR FUNKCJI DO OPTYMALIZACJI")
    print("=" * 50)
    print("1. Funkcja z sekcji 3")
    print("   Dziedzina: [-150, 150]")
    print("   Funkcja z dwoma ekstremami lokalnymi")
    print()
    print("2. Funkcja z sekcji 4")
    print("   Dziedzina: [-3, 12] × [4.1, 5.8]")
    print("   Funkcja z wieloma ekstremami lokalnymi")
    print()
    print("3. Własna funkcja z jedną niewiadomą")
    print("4. Własna funkcja z dwoma niewiadomymi")
    print("=" * 50)

    while True:
        choice = input("Wybierz funkcję (1-4): ").strip()
        match choice:
            case "1":
                return func_section_3, (-150, 150), 1
            case "2":
                return func_section_4, [(-3, 12), (4.1, 5.8)], 2
            case "3":
                return get_custom_function_1d()
            case "4":
                return get_custom_function_2d()
            case _:
                print("Błąd! Wybierz opcję 1-4.")


def get_custom_function_1d():
    """Definiuje własną funkcję z jedną niewiadomą."""
    print("\n" + "=" * 50)
    print("DEFINIOWANIE WŁASNEJ FUNKCJI Z JEDNĄ NIEWIADOMĄ")
    print("=" * 50)
    print("Dostępne funkcje: sin, cos, tan, exp, log, sqrt")
    print("Dostępne stałe: pi, e")
    print("Przykład: x * sin(10 * pi * x) + 1")
    print()
    func_str = input("Podaj wzór funkcji f(x): ")
    left = get_float_input("Podaj lewą granicę dziedziny")
    right = get_float_input("Podaj prawą granicę dziedziny")

    # Tworzymy funkcję lambda z dostępem do funkcji matematycznych
    try:
        # Kontekst z funkcjami matematycznymi
        math_context = {
            'sin': sin, 'cos': cos, 'tan': tan,
            'exp': exp, 'log': log, 'sqrt': sqrt,
            'pi': pi, 'e': math.e, 'abs': abs
        }
        func = eval(f"lambda x: {func_str}", math_context)
        # Test funkcji
        func((left + right) / 2)
        return func, (left, right), 1
    except Exception as e:
        print(f"Błąd w definicji funkcji: {e}")
        print("Powrót do menu wyboru funkcji...")
        return select_function()


def get_custom_function_2d():
    """Definiuje własną funkcję z dwoma niewiadomymi."""
    print("\n" + "=" * 50)
    print("DEFINIOWANIE WŁASNEJ FUNKCJI Z DWOMA NIEWIADOMYMI")
    print("=" * 50)
    print("Dostępne funkcje: sin, cos, tan, exp, log, sqrt")
    print("Dostępne stałe: pi, e")
    print("Przykład: x**2 + y**2 - x*y*sin(pi*x)")
    print()
    func_str = input("Podaj wzór funkcji f(x, y): ")

    x_left = get_float_input("Podaj lewą granicę dziedziny dla x")
    x_right = get_float_input("Podaj prawą granicę dziedziny dla x")
    y_left = get_float_input("Podaj lewą granicę dziedziny dla y")
    y_right = get_float_input("Podaj prawą granicę dziedziny dla y")

    # Tworzymy funkcję lambda z dostępem do funkcji matematycznych
    try:
        # Kontekst z funkcjami matematycznymi
        math_context = {
            'sin': sin, 'cos': cos, 'tan': tan,
            'exp': exp, 'log': log, 'sqrt': sqrt,
            'pi': pi, 'e': math.e, 'abs': abs
        }
        func = eval(f"lambda x, y: {func_str}", math_context)
        # Test funkcji
        test_x = (x_left + x_right) / 2
        test_y = (y_left + y_right) / 2
        func(test_x, test_y)
        return func, [(x_left, x_right), (y_left, y_right)], 2
    except Exception as e:
        print(f"Błąd w definicji funkcji: {e}")
        print("Powrót do menu wyboru funkcji...")
        return select_function()


def get_parameters():
    """Pobiera parametry algorytmu od użytkownika."""
    print("\n" + "=" * 50)
    print("PARAMETRY ALGORYTMU")
    print("=" * 50)

    epochs = get_int_input("Liczba epok", 5000)
    attempts_per_epoch = get_int_input("Liczba iteracji w epoce", 1)
    init_temp = get_float_input("Temperatura początkowa", 500.0)
    alpha = get_float_input("Współczynnik alpha (schładzanie)", 0.999)
    k = get_float_input("Stała k", 0.1)

    return epochs, attempts_per_epoch, init_temp, alpha, k


def display_results(point, value, best_iters, exec_time, dimensions):
    """Wyświetla wyniki działania algorytmu."""
    print("\n" + "=" * 50)
    print("WYNIKI OPTYMALIZACJI")
    print("=" * 50)

    if dimensions == 1:
        print(f"Najlepszy punkt: x = {point[0]:.6f}")
    elif dimensions == 2:
        print(f"Najlepszy punkt: x = {point[0]:.6f}, y = {point[1]:.6f}")
    else:
        print(f"Najlepszy punkt: {[f'{p:.6f}' for p in point]}")

    print(f"Wartość funkcji: f = {value:.6f}")
    print(f"Numer iteracji najlepszego rozwiązania: {best_iters}")
    print(f"Czas wykonania: {exec_time:.3f} ms")
    print("=" * 50)


def main():
    """Główna funkcja programu."""
    print("\n" + "=" * 50)
    print("ALGORYTM SYMULOWANEGO WYŻARZANIA")
    print("=" * 50)

    while True:
        # Wybór funkcji
        func, domain, dimensions = select_function()

        # Pobranie parametrów
        epochs, attempts_per_epoch, init_temp, alpha, k = get_parameters()

        # Utworzenie obiektu algorytmu
        annealing = SimulatedAnnealing(func=func, domain=domain)

        # Uruchomienie algorytmu
        print("\nUruchamianie algorytmu...")
        (point, value, best_iters), exec_time = annealing.run_epochs(
            epochs=epochs,
            attempts_per_epoch=attempts_per_epoch,
            init_temp=init_temp,
            alpha=lambda t: alpha * t,
            k=k
        )

        # Wyświetlenie wyników
        display_results(point, value, best_iters, exec_time, dimensions)

        # Pytanie o ponowne uruchomienie
        print("\nCzy chcesz uruchomić algorytm ponownie?")
        repeat = input("Wpisz 'tak' aby kontynuować lub dowolny klawisz aby zakończyć: ").strip().lower()
        if repeat not in ['tak', 't', 'yes', 'y']:
            print("\nDziękujemy za skorzystanie z programu!")
            break


if __name__ == "__main__":
    main()
