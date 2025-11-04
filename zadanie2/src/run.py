from algorithm import SimulatedAnnealing
from funcs import *


def main():

    # Example from lecture
    annealing_example_1 = SimulatedAnnealing(
        func=example_1,
        domain=(0, 10)
    )
    point, value = annealing_example_1.run_epochs(
        epochs=50,
        attempts_per_epoch=50,
        init_temp=5.0,
        alpha=lambda t: 0.9 * t,
        k=1.0
    )
    print(f'x: {point[0]: .4f}, f(x): {value: .4f}')

    # One of the functions from section 3
    annealing_section_3 = SimulatedAnnealing(
        func=func_section_3,
        domain=(-150, 150)
    )
    point, value = annealing_section_3.run_epochs(
        epochs=5000,
        attempts_per_epoch=1,
        init_temp=500.0,
        alpha=lambda t: 0.999 * t,
        k=0.1
    )
    print(f'x: {point[0]: .4f}, f(x): {value: .4f}')

    # One of the functions from section 4
    annealing_section_4 = SimulatedAnnealing(
        func=func_section_4,
        domain=[(-3, 12), (4.1, 5.8)]
    )
    point, value = annealing_section_4.run_epochs(
        epochs=7000,
        attempts_per_epoch=1,
        init_temp=100.0,
        alpha=lambda t: 0.999 * t,
        k=0.2
    )
    print(f'x: {point[0]: .4f}, y: {point[1]: .4f}, f(x,y): {value: .4f}')


if __name__ == "__main__":
    main()
