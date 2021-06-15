#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# The pipeline module implementing a pipeline
# class, which will be a certain realization of
# the basic "helix" -> set of operations and data
#
# TODO: Add a logger which stores to a "log_<pipeline_name>"
# container

import numpy as np

from tqdm import tqdm

from xileh.core.pipelinedata import xPData as PData
from xileh.core.features import create_features


class xPipeline(object):

    """ The pipeline as a individual realization of
    processing steps and data -> strongly motivated
    by scipy's pipeline
    """

    def __init__(self, name, verbose=False):
        """ Setup with just populating the name for now

        Parameters
        ----------
        name : str
            name of the pipeline
        verbose : bool
            whether or not to print step on .eval()


        """
        self._name = name
        self._steps = []
        self.verbose = verbose

    def add_step(self, step_foo):
        """ Add a processing step

        Parameters
        ----------
        step_foo : tuple (name, function, kwargs)
            function which needs to be able to process
            a PData object and name of the step
        """

        # check that name is not yet used
        assert all([step_foo[0] != t[0] for t in self._steps])

        self._steps.append(step_foo)

    def remove_step(self, name):
        """Remove a step identified by the name

        Parameters
        ----------
        name: str
            name of the step to drop from self._steps

        """

        self._steps = [t for t in self._steps if t[0] != name]

    def eval(self, pdata):
        """ Run all steps in self._steps
        Parameters
        ----------
        pdata : pipelinedata.PData
            The PData object to be processed

        Returns
        -------
            pdata : PData
                Return the pipelined data after running through all steps

        """

        steps_iterator = tqdm(self._steps)
        for step in steps_iterator:
            steps_iterator.set_description(f"Processing step: {step[0]}")
            if self.verbose:
                print(f"Runnning step: {step[0]}")
            foo = step[1]
            kwargs = step[2]
            pdata = foo(pdata, **kwargs)


if __name__ == "__main__":

    tdata = PData(
        data=np.eye(5),
        header={'description': 'Some data description'},
        meta={'mean': 5}
    )

    xpl = xPipeline('testp')
    xpl.add_step(('c22 extract', create_features, {'algo': 'c22'}))
    xpl.eval(tdata)
