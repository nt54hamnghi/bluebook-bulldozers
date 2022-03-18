# place for legacy, i.e. replaced with better implementation, code 


# def loggit(logger: logging.Logger):
#     def inner(fn: Callable[P, R]) -> Callable[P, R]:
#         @wraps(fn)
#         def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
#             result = fn(*args, **kwargs)
#             logger.info(repr(result))
#             return result
#         return wrapper
#     return inner


# def timeit(fn: Callable[P, R]) -> Callable[P, tuple[R, float]]:
#     @wraps(fn)
#     def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[R, float]:
#         start, result, end = time.time(), fn(*args, **kwargs), time.time()
#         time_taken = end - start
#         return result, time_taken
#     return wrapper


# def styleit(fn: Callable[P, None]) -> Callable[P, None]:
#     @wraps(fn)
#     def wrapper(*args, **kwargs):
#         if args:
#             data, *others = args
#         else:
#             data = kwargs.pop("data")
#             others = args
#         styled = set_style(data)
#         fn(styled, *others, **kwargs)
#     return wrapper
