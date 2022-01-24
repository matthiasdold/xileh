from pathlib import Path


def _get_large_files_dir(fname):
    # check for/make large files dir
    p = Path(fname, 'extra')
    p.mkdir(exist_ok=True, parents=True)
    return p


def _create_extra_file_name(large_file_dir, file_name, ext):
    name = large_file_dir.joinpath(f'{file_name}{ext}')
    i = 0
    while name.exists():
        name = large_file_dir.joinpath(f'{file_name}_{i}{ext}')
        i += 1

    return name
