from algorithm import SimulatedAnnealing
from funcs import *


def main():

    annealing_example_1 = SimulatedAnnealing(
        func=example_1,
        domain=(0, 10)
    )

    x, y = annealing_example_1.run_epochs(
        epochs=50,
        attempts_per_epoch=50,
        init_temp=5.0,
        alpha=0.9
    )

    print(f'x: {x: .2f}, y: {y: .2f}')


if __name__ == "__main__":
    main()
