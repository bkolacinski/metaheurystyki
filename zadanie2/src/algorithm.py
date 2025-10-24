class Algorithm:
    init_temp: float
    alpha: float
    iter_num: int
    k_coef: float

    def __init__(self, init_temp: float, alpha: float, iter_num: int, k_coef: float):
        self.init_temp = init_temp
        self.alpha = alpha
        self.iter_num = iter_num
        self.k_coef = k_coef
