from pathlib import Path

from xileh.utils.datahandler.saver.read import get_reader
from xileh.utils.datahandler.saver.prepare import unprepare


def load(res_dir, unprepare_output=False, undo_fname_map=True):
    res_dir = Path(res_dir)
    assert res_dir.exists()

    output = {}

    for file in res_dir.glob('*.*'):  # get files only, ignore the "extra" folder       # noqa
        read = get_reader(file)

        if file.stem.endswith('_epo') and file.suffix == '.fif':
            name = file.stem[:-4]
        else:
            name = file.stem

        output[name] = read(file)

    if undo_fname_map:
        output = undo_fname_mapping(output)

    if unprepare_output:
        return unprepare(output, res_dir=res_dir)
    else:
        return output


def undo_fname_mapping(output):
    """If no fname mapping file found, returns the output"""

    if 'fname_mapping' in output:
        fname_mapping = output.pop('fname_mapping')
        fname_mapping = {new: original for original,
                         new in fname_mapping.items()}  # reverse it
    else:
        return output

    for k in list(output.keys()):
        if k in fname_mapping:
            output[fname_mapping[k]] = output.pop(k)

    return output
