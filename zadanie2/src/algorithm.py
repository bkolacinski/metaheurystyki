from typing import Callable


class Algorithm:
    init_temp: float
    alpha: float
    iter_num: int
    k_coef: float
    func: Callable[[..., float], float]

    def __init__(self, init_temp: float, alpha: float, iter_num: int,
                 k_coef: float, func: Callable[[..., float], float]):
        self.init_temp = init_temp
        self.alpha = alpha
        self.iter_num = iter_num
        self.k_coef = k_coef
        self.func = func

    def run(self):
        stop_condition: bool = False
        # temp: float = self.init_temp
        while not stop_condition:
            for i in range(self.iter_num):
                pass
                # TODO ???
                # wyznaczenie losowo sąsiedniego rozwiązania s’∈
                # N(s)
                #
                # delta: float = func(s_prim) - func(s)
                # if delta < 0:
                #     s = s_prim
                # else:
                #     x = random.uniform(0, 1)
                #     if x < math.exp(-delta / (k_coef * temp)):
                #         s = s_prim
                #
                # TODO ???
                # T = α(T)
