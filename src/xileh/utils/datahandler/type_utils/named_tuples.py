from collections import namedtuple

# dummy for compatibility with the rest of the module -> based of legacy
NAMED_TUPLES = {'Point': namedtuple('Point', 'x y')}


def check_for_unprepared_named_tuple(container):
    """
    For checking if an unprepared named tuple (dict with NamedTuple as a key)
    is present
    in the container at any nested depth.
    :param container: list, tuple, dict, some holder of data
    :return: bool, if dict with NamedTuple found inside at any level
    """
    if isinstance(container, (list, tuple)):
        for element in container:
            check_for_unprepared_named_tuple(element)
    elif isinstance(container, dict):
        if is_unprepared_named_tuple(container):
            return True
        else:
            for k, v in container.items():
                check_for_unprepared_named_tuple(v)
    else:
        # this is assuming the only thing containing named tuples would be
        # lists, tuples and dicts
        return False


def is_outermost_unprepared_tuple(unprepared_tuple):
    ls = []
    for key, value in unprepared_tuple.items():
        contains_named_tuple = check_for_unprepared_named_tuple(value)
        ls.append(contains_named_tuple)
    return not any(ls)


def is_unprepared_named_tuple(container):
    if 'TYPE_CHANGE' in container and container['TYPE_CHANGE'] in NAMED_TUPLES:
        return True
    else:
        return False


def isinstance_namedtuple(obj):
    # https://stackoverflow.com/a/62692640/3087783
    return (
        isinstance(obj, tuple) and
        hasattr(obj, '_asdict') and
        hasattr(obj, '_fields')
    )
