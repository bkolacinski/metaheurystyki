from algorithm import SimulatedAnnealing
from funcs import *


def main():
    # One of the functions from section 3
    annealing_section_3 = SimulatedAnnealing(
        func=func_section_3,
        domain=(-150, 150)
    )
    (point, value, best_iters), _ = annealing_section_3.run_epochs(
        epochs=5000,
        attempts_per_epoch=1,
        init_temp=500.0,
        alpha=lambda t: 0.999 * t,
        k=0.1
    )
    expected_val = 11
    print("\nResult from section 3:")
    print(f'x: {point[0]: .4f}, f(x): {value: .4f}')
    print(f'Expected value: {expected_val}')
    print(f'Difference: {abs(value - expected_val)}')

    # One of the functions from section 4
    annealing_section_4 = SimulatedAnnealing(
        func=func_section_4,
        domain=[(-3, 12), (4.1, 5.8)]
    )
    (point, value, best_iters), _ = annealing_section_4.run_epochs(
        epochs=7000,
        attempts_per_epoch=1,
        init_temp=100.0,
        alpha=lambda t: 0.999 * t,
        k=0.2
    )
    expected_val = 38.850294
    print("\nResult from section 4:")
    print(f'x: {point[0]: .4f}, y: {point[1]: .4f}, f(x,y): {value: .4f}')
    print(f'Expected value: {expected_val}')
    print(f'Difference: {abs(value - expected_val)}')


if __name__ == "__main__":
    main()
