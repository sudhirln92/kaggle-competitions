import time
from functools import wraps


def logging_time(func):
    @wraps(func)
    def logged(*args, **kwargs):
        ex_start = time.time()
        res = func(*args, *kwargs)
        elapsed_time = time.time() - ex_start
        print(f"{func.__name__} time elapsed-----:\t{elapsed_time:.5f}")
        return res

    return logged