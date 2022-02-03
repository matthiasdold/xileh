# some snippets no longer being used


#
# def non_serializeable_in_iterable(iter):
#     """ Go over all entries in an list or tuple, and check if
#     any contains a non_serializeable_type
#     """
#
#     # Note as a set can only contain hashable types, we do not need to check
#     # for sets
#     return any([isinstance(v, tuple(non_serializeable_types.keys()))
#                 if not isinstance(v, (list, tuple))
#                 else non_serializeable_in_iterable(v)
#                 for v in iter])
#
#
# def dict_in_iterable(iter):
#     """ Go over all entries in an list or tuple, and check if
#     any contains a dict
#     """
#
#     # Note as a set can only contain hashable types, we do not need to check
#     # for sets
#     return any([isinstance(v, dict) if not isinstance(v, (list, tuple))
#                 else dict_in_iterable(v)
#                 for v in iter])
#
#

# =============================================================================
# Data layout
# =============================================================================


def get_savers_layout(data):
    """ Iterate over a data object and get a mapping layout with saver funcs

    Parameters
    ----------
    data : dict
        key value pairs to store in a container folder


    Returns
    -------
    saver_layout : dict
        key value pairs for data entities names and their savers

    """

    saver_layout = {}
    for k, v in data.items():
        saver_layout[k] = get_saver(v)

    return saver_layout


def get_saver(value):
    """ For a given value/object, get the appropriate saver"""

    if isinstance(value, dict):
        ret = get_savers_layout(value)
    elif isinstance(value, tuple(non_serializeable_types)):
        ret = non_serializeable_types[type(value)]
    elif isinstance(value, (list, tuple, set)):
        ret = [get_saver(d) for d in value]
    else:
        ret = value

    return ret


# def test_dict_in_iterable():
#     a = [1, dict(), 3, "test"]
#     b = [1, "as", [dict(a=2), 5]]
#     c = [1, (2, "test", [dict(b=3), 5]), "as"]
#     d = [1, (2, "test", [6, 5]), "as"]
#
#     assert dict_in_iterable(a), "Missed a dict"
#     assert dict_in_iterable(b), "Missed a dict"
#     assert dict_in_iterable(c), "Missed a dict"
#     assert not dict_in_iterable(d), "No dict in test data but found one"

#
# def test_non_serializeable_type_in_iterable():
#     a = [1, pd.DataFrame(), 3, "test"]
#     b = [1, "as", [np.ones(3), 5]]
#     c = [1, (2, "test", [np.zeros(5), 5]), "as"]
#     d = [1, (2, "test", [6, 5]), "as"]
#
#     assert non_serializeable_in_iterable(a), "Missed a non_serializeable"
#     assert non_serializeable_in_iterable(b), "Missed a non_serializeable"
#     assert non_serializeable_in_iterable(c), "Missed a non_serializeable"
#     assert not non_serializeable_in_iterable(d), "No non_serializeable in "\
#         "test data but found one"
#

