#!/usr/bin/env python
#
# A rich-backed tqdm progress bar that plays nicely with logging.

import os
import warnings

from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm_rich as _tqdm_rich


class tqdm(_tqdm_rich):
    """tqdm wrapper with a ``show_bar`` kwarg and DISABLE_PROGRESS_BAR env var support.

    The bar is suppressed when either:
    - ``show_bar=False`` is passed, or
    - the environment variable ``DISABLE_PROGRESS_BAR`` is set to any non-empty value.
    """

    def __init__(self, *args, show_bar: bool = True, **kwargs):
        if not show_bar or os.environ.get("DISABLE_PROGRESS_BAR"):
            kwargs["disable"] = True
        # rich support is flagged experimental by tqdm itself; the warning is
        # not actionable for users of this wrapper, so silence it locally.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", TqdmExperimentalWarning)
            super().__init__(*args, **kwargs)
