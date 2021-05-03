#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 2021-03-25
#
# Each processing step might have certain requirements for the data
# in the pipeline data structure - here are methods for checking
# and logging

import functools


def check_requirements(func):
    """ For a given function check which information is required
    and whether it is contained in the provided arg (usually a
    pipeline data)

    Parameters
    ----------
    func : function
        any function processing a core.pipelinedata.PData instance
        as first element

    Returns
    -------
        wfunc : function
            wrapped function which determines data completeness
            on each call

    """
    pass
