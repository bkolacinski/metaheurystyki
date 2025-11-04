from math import sin, pi


def example_1(x: float) -> float:
    return 3 * sin(pi * x / 5) + sin(pi * x)


def func_section_3(x: float) -> float:
    if -105 < x < -95:
        return 10 - 2 * abs(x + 100)
    elif 95 < x < 105:
        return 11 - 2.2 * abs(x - 100)
    return 0


def func_section_4(x: float, y: float) -> float:
    return 21.5 + x * sin(4 * pi * x) + y * sin(20 * pi * y)
