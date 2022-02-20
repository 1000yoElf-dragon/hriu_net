# Auxiliary functions and classes

def iterate(x):
    """

    :param x: iterable or non-iterable
    :return: if x is iterable returns iterator through x, otherwise iterator with only one element x
    Be careful with strings
    """
    try:
        yield from iter(x)
    except TypeError:
        yield x


def prod(x):
    """

    :param x: iterable or number
    :return: if x is iterable returns product of it's elements (int(1) when empty), otherwise x
    """
    res = 1
    for v in iterate(x):
        res *= v
    return res


def tuple_n(dims, var, var_name='Shape parameter'):
    """
    if 'len(var)' equal 'dims' returns 'tuple(var)'
    if var is single integer returns tuple with 'dims' elements
    otherwise raise error

    :param dims: desired number of elements in resulting tuple
    :param var: convertible to integer or collection of integers having 'len'
    :param var_name: name to display if error occurred
    :return: tuple with dims elements
    """
    try:
        if len(var) == dims: return tuple(map(int, var))
        error_type = ValueError
    except TypeError:
        try:
            return (int(var),)*dims
        except (TypeError, ValueError, OverflowError) as error:
            error_type = type(error)
    except (ValueError, OverflowError) as error:
        error_type = type(error)
    raise error_type(
            f"'{var_name}' must be {dims}-tuple or integer, actually is {type(var)}: {var}"
            )


class Absolute:
    """
    Emulator of collection containing everything
    """
    def __contains__(self, item): return True
