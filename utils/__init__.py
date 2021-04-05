def eq_tol(a: float, b: float, tol=1e-3):
    assert tol >= 0
    return abs(a - b) <= tol


def is_iter(x):
    try:
        iterator = iter(x)
        return True
    except TypeError:
        return False


# iterable

def flatten(*iterates):
    for i, x in enumerate(iterates):
        assert is_iter(x), f'Element at {i + 1} position is not iterable'

    return [x for sub in iterates for x in sub]
