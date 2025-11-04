import matplotlib.pyplot as plt
import numpy as np

from funcs import *


def main():
    x = np.linspace(-150, 150, 3000)
    y = [func_section_3(xi) for xi in x]

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.savefig('../sprawozdanie/figures/func_section_3.png')


if __name__ == "__main__":
    main()
