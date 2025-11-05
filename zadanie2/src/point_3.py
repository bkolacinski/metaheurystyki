import matplotlib.pyplot as plt
import numpy as np

from algorithm import SimulatedAnnealing
from funcs import func_section_3


def draw_plots(x, results, title):
    times = [result[0] for result in results]
    iters = [result[1] for result in results]
    diffs = [result[2] for result in results]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))
    fig.suptitle(f'Dependence of parameters on {title}', fontsize=16)

    axes[0].plot(x, times, 'b-', linewidth=2)
    axes[0].set_xlabel(f'{title}')
    axes[0].set_ylabel('Time [s]')
    axes[0].set_title(f'{title} vs Time')
    axes[0].grid(True)

    axes[1].plot(x, iters, 'g-', linewidth=2)
    axes[1].set_xlabel(f'{title}')
    axes[1].set_ylabel('Number of iterations')
    axes[1].set_title(f'{title} vs Number of iterations')
    axes[1].grid(True)

    axes[2].plot(x, diffs, 'r-', linewidth=2)
    axes[2].set_xlabel(f'{title}')
    axes[2].set_ylabel('Difference from global max')
    axes[2].set_title(f'{title} vs Difference from global max')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f'../sprawozdanie/figures/{title.lower()}_overview.png')
    plt.show()


def test_parameter(algorithm: SimulatedAnnealing,
                   length: int,
                   temperatures=None,
                   alphas=None,
                   epochs=None) -> list[tuple[float, int, float]]:
    results = []
    for i in range(length):
        sum_time: float = 0.0
        sum_best_iters: int = 0
        sum_difference: float = 0.0
        for _ in range(5):
            (point, value, best_iters), time = algorithm.run_epochs(
                epochs=(5000 if epochs is None else epochs[i]),
                attempts_per_epoch=1,
                init_temp=(500.0 if temperatures is None else temperatures[i]),
                alpha=lambda t: (0.999 if alphas is None else alphas[i]) * t,
                k=0.1
            )
            sum_time += time
            sum_best_iters += best_iters
            sum_difference += abs(value - 11)

        results.append((sum_time / 5, sum_best_iters // 5, float(sum_difference / 5)))

    return results


def main():
    annealing_section_3 = SimulatedAnnealing(
        func=func_section_3,
        domain=(-150, 150)
    )

    # temperatures = np.linspace(1, 1000, 200)
    # temperatures_results = test_parameter(
    #     algorithm=annealing_section_3,
    #     length=len(temperatures),
    #     temperatures=temperatures
    # )
    # draw_plots(temperatures, temperatures_results, 'Temperature')

    # alphas = np.linspace(0.8, 1, 100)
    # alphas_results = test_parameter(
    #     algorithm=annealing_section_3,
    #     length=len(alphas),
    #     alphas=alphas
    # )
    # draw_plots(alphas, alphas_results, 'Alpha')

    epochs = np.linspace(100, 10000, 200, dtype=int)
    epochs_results = test_parameter(
        algorithm=annealing_section_3,
        length=len(epochs),
        epochs=epochs
    )
    draw_plots(epochs, epochs_results, 'Epochs')


if __name__ == "__main__":
    main()
