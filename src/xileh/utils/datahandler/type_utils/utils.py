def is_unprepared_type_dict(x):
    if isinstance(x, dict) and 'TYPE_CHANGE' in x:
        return True
    else:
        return False


def _is_outermost(x, fn):
    ls = []
    if isinstance(x, dict):
        x = x.values()
    elif isinstance(x, (tuple, list)):
        pass
    else:
        return True

    for v in x:
        check = _is_outermost(v, fn=fn)
    # for k, v in x.items():
    #     check = _is_outermost(v, fn=fn)
        ls.append(check)

    return not any(ls)


def is_outermost_unprepared_type_dict(x):
    return _is_outermost(x, fn=is_unprepared_type_dict)

# After using datetime.datetime.now() for the meas_date in the mne fake test
# data, the generator approach broke --> interpreter dying with error 139
# --> let's use simple iterations with early stopping


def contains_tuple(obj):
    chk = False
    if isinstance(obj, tuple):
        chk = True
    elif isinstance(obj, list):
        for elm in obj:
            chk |= contains_tuple(elm)
            if chk:
                break
    elif isinstance(obj, dict):
        for elm in obj.values():
            chk |= contains_tuple(elm)
            if chk:
                break

    return chk
