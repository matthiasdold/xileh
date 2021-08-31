#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 20210817
#
# This module includes monitoring utilities as e.g. used for monitoring
# multiple pipelines being executed in either subprocesses and/or remotely

import re
import time

from tqdm import tqdm


class PipelineMonitor(object):

    """
    PipelineMonitor checking on the log files of pipeline(s) being
    executed.
    """

    def __init__(self, logfiles, check_freq=2, names=[]):
        """
        Parameters
        ----------
        log_files : list of Paths
            path to the log files to monitor
        check_freq : float
            number of seconds to sleep between checking the files
        names : list
            a list of str used as names for the tqdm bars, empty use
            the log file stems
        """
        self._logfiles = logfiles
        self._pl_names = [f.stem for f in logfiles] if names == [] else names
        self._check_freq = check_freq
        self.step_state = [0] * len(logfiles)
        self.update_bar_at_idx = [0] * len(logfiles)
        self._bar_totals_set = [False] * len(logfiles)

    def _init_bars(self):
        """ Initialize bars with just one step"""

        self.bars = []
        for pos, name in enumerate(self._pl_names):
            bar = tqdm(
                desc=f"{name} - waiting for first step to finish",
                total=2,
                position=pos,   # position for ranking multiple bars
                leave=True,     # always at the bottom of the prompt
            )

            self.bars.append(bar)

    def _bar_i_set_total(self, i, total):
        name = self._pl_names[i]
        bar = self.bars[i]
        bar.desc = f"{name}"
        bar.total = int(total)
        self._bar_totals_set[i] = True

    def update_bars(self):
        for i, update_val in enumerate(self.update_bar_at_idx):

            if update_val > 0:
                self.bars[i].update(update_val)
                self.update_bar_at_idx[i] = 0

    def check_logs(self):
        """ Read the logs and check the last finished step """
        # tqdm.write(
        #     f"Checking logfiles - {self._logfiles} - time:"
        #     f" {time.strftime('%Y%m%d_%H%M%S')}")
        for i, lf in enumerate(self._logfiles):
            text = open(lf, 'r').read()
            step_info = re.findall(r'Finished step (\d*)/(\d*)', text)
            # tqdm.write(f"Step info is: {step_info}")

            # if at least one step finished, update the state
            if step_info != []:

                # complete the bars now that we know the max number of steps
                if not all(self._bar_totals_set):
                    self._bar_i_set_total(i, step_info[0][1])

                last_finished = int(step_info[-1][0])
                if last_finished > self.step_state[i]:
                    self.update_bar_at_idx[i] = (last_finished
                                                 - self.step_state[i])
                    self.step_state[i] = last_finished

    def show(self):
        self._init_bars()

        while 1:
            self.check_logs()
            self.update_bars()
            time.sleep(self._check_freq)

            # check if all are finished
            totals = [b.total for b in self.bars]

            if all([s >= smax for s, smax in zip(self.step_state, totals)]):
                break

    def __del__(self):
        for b in self.bars:
            b.close()


if __name__ == '__main__':

    from pathlib import Path

    logfs = [
        Path('/tmp/testlog2.log'),
        Path('/tmp/testlog3.log'),
    ]

    from multiprocessing import Pool
    import random

    def print_to_log(logf, msg='test', iterations=10, sleep_for=2):
        for i in range(iterations):
            tqdm.write(msg, file=open(logf, 'a'))
            tqdm.write(f"Finished step {i + 1}/{iterations}",
                       file=open(logf, 'a'))
            time.sleep(random.uniform(0.5, sleep_for))

    pool = Pool(len(logfs))
    for i in range(len(logfs)):
        pool.apply_async(print_to_log, args=(logfs[i],))

    # wait for files to be written for the first time
    time.sleep(2)

    logfs = [Path('/tmp/nemo_eval/testpipeline.log'),
             Path('/tmp/nemo_eval/testpipeline2.log')]

    m = PipelineMonitor(logfiles=logfs)
    m.show()
    m.__del__()

    pool.close()
    pool.join()

    for f in logfs:
        if f.exists():
            f.unlink()
