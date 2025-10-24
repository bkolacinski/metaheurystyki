from math import sin, pi


def func_section_3(x: float) -> float:
    if x > -105 and x < -95:
        return 10 - 2 * abs(x + 100)
    elif x > 95 and x < 105:
        return 11 - 2.2 * abs(x - 100)
    return 0


def func_section_4(x: float, y: float) -> float:
    return 21.5 + x * sin(4 * pi * x) + y * sin(20 * pi * y)
