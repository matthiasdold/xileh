#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# The pipeline module implementing a pipeline
# class, which will be a certain realization of
# the basic "helix" -> set of operations and data
#
# The pipeline implementations

import hashlib
import numpy as np

from xileh.utils.progress import tqdm
from xileh.core.pipelinedata import xData
from xileh.utils.logger import PlainLogger


class xPipeline(object):
    """The pipeline as a individual realization of
    processing steps and data -> strongly motivated
    by scipy's pipeline
    """

    def __init__(
        self,
        name: str,
        verbose: bool = False,
        silent: bool = False,
        log_eval: bool = False,
    ):
        """Setup with just populating the name for now

        Parameters
        ----------
        name : str
            name of the pipeline
        verbose : bool (False)
            whether or not to print step on .eval()
        silent : bool (False)
            if silent, do not print a progress bar. This is used if timing
            is critical
        log_eval : bool (False)
            whether or not to log the evaluation
        """
        self._name = name
        self._steps = []
        self._steps_hash = self._compute_steps_hash()
        self.verbose = verbose
        self._logger = PlainLogger(name + ".log")
        self._log_eval = log_eval
        self._silent = silent

    def __repr__(self):
        """Show name of repl call"""
        return (
            super().__repr__()
            + f"\nPipeline name: {self._name}"
            + "\nSteps: "
            + self.pretty_print_get_steps()
        )

    def _compute_steps_hash(self) -> str:
        parts = [
            f"{s[0]}:{s[1].__name__}:{str(s[2]) if len(s) > 2 else '{}'}"
            for s in self._steps
        ]
        return hashlib.md5("|".join(parts).encode()).hexdigest()

    def _update_hash(self):
        self._steps_hash = self._compute_steps_hash()

    @property
    def steps_hash(self) -> str:
        return self._steps_hash

    def pretty_print_get_steps(self):
        step_names = [f"'{s[0]}'" for s in self._steps]
        return "\n\t-> " + "\n\t-> ".join(step_names)

    def check_step_foo(self, step_foo):
        """Check the tuple formulating a step function

        Parameters
        ----------
        step_foo : tuple (name, function, kwargs)
            name of the step, function which needs to be able to process
            a xData object and (optional) kwargs for this function

        Returns
        -------
        step_foo : tuple (name, function, kwargs)
            tuple as input but with kwargs={} extended in case step_foo only
            contains (name, function)
        """

        assert len(step_foo) in (2, 3), (
            "A step must be a (name, function) or (name, function, kwargs) "
            f"tuple, got {len(step_foo)} elements: {step_foo}"
        )
        assert callable(step_foo[1]), (
            f"The second step element must be the callable processing function,"
            f" got {type(step_foo[1]).__name__}: {step_foo[1]}"
        )
        if len(step_foo) == 3:
            assert isinstance(step_foo[2], dict), (
                "The third step element must be a kwargs dict, got "
                f"{type(step_foo[2]).__name__}: {step_foo[2]}"
            )
        else:
            step_foo = tuple(list(step_foo) + [{}])

        return step_foo

    def add_step(self, step_foo: tuple):
        """Add a processing step

        Parameters
        ----------
        step_foo : tuple (name, function, kwargs)
            name of the step, function which needs to be able to process
            a xData object and (optional) kwargs for this function
        """

        # check that name is not yet used
        assert all([step_foo[0] != t[0] for t in self._steps]), (
            "Name already in pipeline steps names"
        )
        step_foo = self.check_step_foo(step_foo)

        self._steps.append(step_foo)
        self._update_hash()

    def add_steps(self, *steps):
        """Add a processing steps

        Parameters
        ----------
        steps : tuples (name, function, kwargs)
            name of the step, function which needs to be able to process
            a xData object and (optional) kwargs for this function
        """
        for step in steps:
            self.add_step(step)

    def remove_step(self, name: str):
        """Remove a step identified by the name

        Parameters
        ----------
        name: str
            name of the step to drop from self._steps

        """

        self._steps = [t for t in self._steps if t[0] != name]
        self._update_hash()

    def remove_steps(self, names: list[str]):
        """Remove multiple steps identified by their names

        Parameters
        ----------
        names: list[str]
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

        step, idx = [(t, i) for i, t in enumerate(self._steps) if t[0] == name][0]

        return step, idx

    def __len__(self):
        return len(self._steps)

    def __getitem__(self, key):
        selected = self._steps[key]
        # integer index -> single step, wrap in list for uniform handling
        if isinstance(key, int):
            selected = [selected]
        sub = xPipeline(
            self._name,
            verbose=self.verbose,
            silent=self._silent,
            log_eval=self._log_eval,
        )
        sub._steps = list(selected)
        return sub

    @property
    def steps(self):
        return self._steps

    @steps.setter
    def steps(self, steps):
        self._steps = [self.check_step_foo(s) for s in steps]
        self._update_hash()

    def replace_step(self, name, step_foo):
        """Replace a function given its name

        Parameters
        ----------
        name : str
            step name i.e first value of the step tuple
        step_foo : tuple (name, function, kwargs)
            name of the step, function which needs to be able to process
            a xData object and (optional) kwargs for this function

        """

        step, idx = self.get_step(name)
        step_foo = self.check_step_foo(step_foo)

        self._steps[idx] = step_foo
        self._update_hash()

    def set_step_kwargs(self, name, **kwargs):
        """Update the kwargs set for a particular step"""

        step, idx = self.get_step(name)

        # tuple is unmuteable, but the dictionary at step[2] is muteable
        step[2].update(kwargs)
        self._update_hash()

    def eval(self, pdata: xData):
        """Run all steps in self._steps
        Parameters
        ----------
        pdata : pipelinedata.xData
            The xData object to be processed

        Returns
        -------
            pdata : xData
                Return the pipelined data after running through all steps

        """
        if self._log_eval:
            self._logger.info(
                f"Evaluating pipeline <{self.__hash__()}> with"
                f" data <{pdata.__hash__()}>"
            )

        if self._silent:
            for step in self._steps:
                # check if early_stop is set and breakout
                if "early_stop" in pdata.header.keys() and pdata.header["early_stop"]:
                    break

                foo = step[1]
                kwargs = step[2] if len(step) > 2 else {}
                pdata = foo(pdata, **kwargs)
        else:
            steps_iterator = tqdm(self._steps, position=0, leave=True)
            for i, step in enumerate(steps_iterator):
                # check if early_stop is set and breakout
                if "early_stop" in pdata.header.keys() and pdata.header["early_stop"]:
                    break

                steps_iterator.set_description(f"Processing step: {step[0]}")

                kwargs = step[2] if len(step) > 2 else {}

                n_of_m = f"{i + 1}/{len(self._steps)}"
                msg = f"Eval step {n_of_m}: {step[0]} with kwargs = {kwargs}"

                if self.verbose:
                    tqdm.write(msg)
                if self._log_eval:
                    self._logger.info(msg)

                foo = step[1]
                pdata = foo(pdata, **kwargs)

                if self._log_eval:
                    self._logger.info(f"Finished step {n_of_m}: {step[0]}")


if __name__ == "__main__":
    tdata = xData(
        data=np.eye(5),
        header={"description": "Some data description"},
        meta={"mean": 5},
        name="testing_container",
    )

    def create_features(pdata):
        print("Doing something")
        return pdata

    xpl = xPipeline("testp", verbose=True)

    xpl.add_steps(
        ("test1", create_features, {}),
        ("test2", create_features),
    )

    xpl.eval(tdata)
