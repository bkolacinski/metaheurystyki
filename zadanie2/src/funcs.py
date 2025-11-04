from math import sin, pi
from time import perf_counter


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        return result, (end_time - start_time) * 1000  # time in milliseconds

    return wrapper


def func_section_3(x: float) -> float:
    if -105 < x < -95:
        return 10 - 2 * abs(x + 100)
    elif 95 < x < 105:
        return 11 - 2.2 * abs(x - 100)
    return 0


def func_section_4(x: float, y: float) -> float:
    return 21.5 + x * sin(4 * pi * x) + y * sin(20 * pi * y)
