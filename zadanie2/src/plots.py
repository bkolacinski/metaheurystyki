import matplotlib.pyplot as plt
import numpy as np

from funcs import *
from algorithm import SimulatedAnnealing


def draw_plot_section_3():
    x = np.linspace(-150, 150, 3000)
    y = [func_section_3(xi) for xi in x]

    (point, value, best_iters), _ = SimulatedAnnealing(
        func=func_section_3,
        domain=(-150, 150)
    ).run_epochs(
        epochs=5000,
        attempts_per_epoch=1,
        init_temp=500.0,
        alpha=lambda t: 0.999 * t,
        k=0.1
    )

    plt.plot(x, y)
    plt.scatter(point[0], value, color='red', zorder=10, alpha=0.5,
                label=f'Global maximum at x=100, f(x)=11')
    plt.scatter(100, 11, color='purple', zorder=10, alpha=0.5,
                label=f'Found maximum at x={point[0]:.2f}, f(x)={value:.2f}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    # plt.savefig('../sprawozdanie/figures/func_section_3.png')
    plt.show()


def draw_plot_section_4():
    x = np.linspace(-3, 12, 150)
    y = np.linspace(4.1, 5.8, 170)
    xx, yy = np.meshgrid(x, y)

    zz = np.array([[func_section_4(xi, yi) for xi, yi in zip(x_row, y_row)]
                  for x_row, y_row in zip(xx, yy)])

    (point, value, best_iters), _ = SimulatedAnnealing(
        func=func_section_4,
        domain=[(-3, 12), (4.1, 5.8)]
    ).run_epochs(
        epochs=7000,
        attempts_per_epoch=1,
        init_temp=100.0,
        alpha=lambda t: 0.999 * t,
        k=0.2
    )

    contour = plt.contourf(xx, yy, zz, levels=50, cmap='plasma')
    plt.colorbar(contour)

    plt.scatter(point[0], point[1], color='olivedrab', zorder=11,
                edgecolors='white', linewidths=1.5, s=50,
                label=f'Found maximum at ({point[0]:.2f}, {point[1]:.2f}), '
                      f'f(x, y)={value:.2f}')
    plt.scatter(11.625545, 5.7250444, color='orchid', zorder=10,
                edgecolors='white', linewidths=1.5, s=50,
                label=f'Global maximum at (11.63, 5.73), f(x, y)=38.85')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('../sprawozdanie/figures/func_section_4.png')
    plt.show()


def main():
    draw_plot_section_3()
    draw_plot_section_4()


if __name__ == "__main__":
    main()
