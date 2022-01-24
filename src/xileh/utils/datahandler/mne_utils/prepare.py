import mne

from xileh.utils.datahandler.saver.utils import (
    _get_large_files_dir, _create_extra_file_name)


def prepare_epochs_array(epochs_array, mode='json', key=None, fname=None):

    if epochs_array.times.flags.writeable:
        epochs_array.times.setflags(write=False)

    if not isinstance(epochs_array.drop_log, tuple):
        epochs_array.drop_log = tuple(map(tuple, epochs_array.drop_log))

    return epochs_array


def _save_mne_to_fif(data, mode='', key='', fname=''):

    type_suffix_map = {
        'epo_fif': (mne.epochs.EpochsFIF, '_epo.fif'),
        'epo': (mne.epochs.Epochs, '_epo.fif'),
        'ica': (mne.preprocessing.ica.ICA, '_ica.fif'),
        'raw': (mne.io.BaseRaw, '_raw.fif'),
        'epochs_array': (mne.EpochsArray, '_epo.fif')
    }

    # loop was the keys might just be parent classes from which inherited
    sfx = ''
    for v in type_suffix_map.values():
        if isinstance(data, v[0]):
            sfx = v[1]
            if v[0] == mne.EpochsArray:
                data = prepare_epochs_array(data)

    assert sfx != '', f"Unknonw data type for {data=}, " \
        f"implemented are {list(type_suffix_map.keys())}"

    p = _get_large_files_dir(fname)
    name = _create_extra_file_name(p, key, sfx)

    # calling save even without overwrite=True should not lead to an error
    # as file names are dynamically incremented via _create_extra_file_name
    data.save(name)

    return dict(path=str(name.relative_to(fname)), TYPE_CHANGE='FIF')
