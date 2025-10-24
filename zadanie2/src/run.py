from .algorithm import Algorithm
from .funcs import *


def main():
    def alpha_func(t: float) -> float:
        return 0.999 * t
        
    algorithm_section_3 = Algorithm(init_temp=500, alpha=alpha_func,
                                    k_coef=0.1, iter_num=3000,
                                    func=func_section_3)


if __name__ == "__main__":
    main()
