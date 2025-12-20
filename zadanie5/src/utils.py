def timer(func):
    from time import perf_counter

    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        return result, (end_time - start_time) * 1000

    return wrapper
