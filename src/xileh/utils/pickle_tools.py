#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 20210818
#
# Tools for interaction of xileh objects with pickle / dill

import os

from pathlib import Path


def pipeline_dependencies(pl):
    """
    For a given pipeline get the files for its steps

    THE FOLLOWING WORKING ASSUMPTION IS MADE:
        - functions within a given pipeline have their dependencies in
          a single module and those pipelines are definded in the root
          directory.
        - all other dependencies would be present in the python environment
          where a potential unpack is happening
        - current script is always copied over which will hold functions
          defined under __main__ . We use dill instead of pickle as this
          should take care of dependencies

    Parameters
    ----------
    pl : xPipeline

    Returns
    -------
    non_venv_paths : list
        list of paths to __module__'s being used in the pipeline steps functions
        which do not have a 'site-packages' in their paths
    """

    # functions are either having their scope as seen from __main__ defined
    # or are part of main themselves. dill will take care of __main__
    # as for the rest we just copy every py file
    foo_paths = [stp[1].__globals__['__file__']
                 if stp[1].__module__ != '__main__'
                 else os.path.abspath(os.path.curdir)
                 for stp in pl._steps]

    # for paths within a 'site-packages' folder --> assume their are installed
    # to the unpacking environment as well
    non_venv_paths = [Path(p) for p in foo_paths if 'site-packages' not in p]

    return non_venv_paths


def test_print(pdata):
    print("Testing")


if __name__ == "__main__":

    # for testing purposes
    from xileh import xPipeline
    from xileh.utils.nemo_eval import eval_pl

    pl = eval_pl
    pl.add_step(('testprint', test_print, {}))
