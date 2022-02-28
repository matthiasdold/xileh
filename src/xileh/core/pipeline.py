#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# The pipeline module implementing a pipeline
# class, which will be a certain realization of
# the basic "helix" -> set of operations and data
#
# The pipeline implementations

import numpy as np

from tqdm import tqdm

from xileh.core.pipelinedata import xPData as PData
from xileh.utils.logger import PlainLogger


class xPipeline(object):

    """ The pipeline as a individual realization of
    processing steps and data -> strongly motivated
    by scipy's pipeline
    """

    def __init__(self, name, verbose=False, log_eval=False):
        """ Setup with just populating the name for now

        Parameters
        ----------
        name : str
            name of the pipeline
        verbose : bool
            whether or not to print step on .eval()
        log_eval : bool
            whether or not to log the evaluation
        """
        self._name = name
        self._steps = []
        self.verbose = verbose
        self._logger = PlainLogger(name + ".log")
        self._log_eval = log_eval

    def __repr__(self):
        """ Show name of repl call """
        return (super().__repr__() + f"\nPipeline name: {self._name}"
                + "\nSteps: " + self.pretty_print_get_steps())

    def pretty_print_get_steps(self):
        step_names = [f"'{s[0]}'" for s in self._steps]
        return '\n\t-> ' + '\n\t-> '.join(step_names)

    def check_step_foo(self, step_foo):
        """ Check the tuple formulating a step function

        Parameters
        ----------
        step_foo : tuple (name, function, kwargs)
            name of the step, function which needs to be able to process
            a PData object and (optional) kwargs for this function

        Returns
        -------
        step_foo : tuple (name, function, kwargs)
            tuple as input but with kwargs={} extended in case step_foo only
            contains (name, function)
        """

        if len(step_foo) < 3 and not isinstance(step_foo[-1], dict):
            step_foo = tuple(list(step_foo) + [{}])

        return step_foo

    def add_step(self, step_foo):
        """ Add a processing step

        Parameters
        ----------
        step_foo : tuple (name, function, kwargs)
            name of the step, function which needs to be able to process
            a PData object and (optional) kwargs for this function
        """

        # check that name is not yet used
        assert all([step_foo[0] != t[0] for t in self._steps]), "Name already"\
            " in pipeline steps names"
        step_foo = self.check_step_foo(step_foo)

        self._steps.append(step_foo)

    def add_steps(self, *steps):
        """ Add a processing steps

        Parameters
        ----------
        steps : tuples (name, function, kwargs)
            name of the step, function which needs to be able to process
            a PData object and (optional) kwargs for this function
        """
        for step in steps:
            self.add_step(step)

    def remove_step(self, name):
        """Remove a step identified by the name

        Parameters
        ----------
        name: str
            name of the step to drop from self._steps

        """

        self._steps = [t for t in self._steps if t[0] != name]

    def remove_steps(self, names):
        """Remove a step identified by the name

        Parameters
        ----------
        name: list[str]
            names of the steps to drop from self._steps

        """
        for name in names:
            self.remove_step(name)

    def get_step(self, name):
        """Get step by name

        Parameters
        ----------
        name : str
            step name i.e first value of the step tuple


        Returns
        -------
        step : tuple (name, function, kwargs)
            the selected processing step
        idx : int
            index of step within self._steps
        """

        step, idx = [(t, i) for i, t in enumerate(self._steps)
                     if t[0] == name][0]

        return step, idx

    @property
    def steps(self):
        return self._steps

    @steps.setter
    def steps(self, steps):
        self._steps = steps

    def replace_step(self, name, step_foo):
        """ Replace a function given its name

        Parameters
        ----------
        name : str
            step name i.e first value of the step tuple
        step_foo : tuple (name, function, kwargs)
            name of the step, function which needs to be able to process
            a PData object and (optional) kwargs for this function

        """

        step, idx = self.get_step(name)
        step_foo = self.check_step_foo(step_foo)

        self._steps[idx] = step_foo

    def set_step_kwargs(self, name, **kwargs):
        """ Update the kwargs set for a particular step """

        step, idx = self.get_step(name)

        # tuple is unmuteable, but the dictionary at step[2] is muteable
        step[2].update(kwargs)

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
        if self._log_eval:
            self._logger.info(f"Evaluating pipeline <{self.__hash__()}> with"
                              f" data <{pdata.__hash__()}>")

        steps_iterator = tqdm(self._steps, position=0, leave=True)
        for i, step in enumerate(steps_iterator):
            steps_iterator.set_description(f"Processing step: {step[0]}")

            n_of_m = f"{i + 1}/{len(self._steps)}"
            msg = f"Runnning step {n_of_m}: {step[0]} with kwargs = {step[2]}"

            if self.verbose:
                tqdm.write(msg)
            if self._log_eval:
                self._logger.info(msg)

            foo = step[1]
            kwargs = step[2]
            pdata = foo(pdata, **kwargs)

            if self._log_eval:
                self._logger.info(f"Finished step {n_of_m}: {step[0]}")


if __name__ == "__main__":

    tdata = PData(
        data=np.eye(5),
        header={'description': 'Some data description'},
        meta={'mean': 5},
        name='testing_container'
    )

    def create_features(pdata):
        # do something
        return pdata

    xpl = xPipeline('testp', verbose=True)
    xpl.add_step(('c22 extract', create_features, {'algo': 'c22'}))
    xpl.add_step(('c22 extract2', create_features))
    xpl.add_step(('c22 extract3', create_features, {'algo': 'c22'}))

    xpl.add_steps(
        ('test1', create_features, {}),
        ('test2', create_features),
    )

    xpl.eval(tdata)
