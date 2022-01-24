from mne.utils._bunch import NamedInt, NamedFloat

from xileh.utils.datahandler.saver.serialize import (
    _prepare_named_digits_for_serialization)


def _prepare_info_chs_named_ints(epochs_array, mode='json'):
    for i, ch in enumerate(epochs_array.info['chs']):
        for k, v in ch.items():
            if isinstance(v, (NamedInt, NamedFloat)):
                epochs_array.info['chs'][i][k] = (
                    _prepare_named_digits_for_serialization(v, key=k,
                                                            mode=mode)
                )

    return epochs_array
