#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 2021-03-25
#
# Definition of the pipeline logger
#
# Target for usage: Apply the logger as a decorator
#
# NOTE: We also implement a file logger with plain print function -->
# reason being that the logging.Logger cannot be pickled, which would be
# necessary to evaluate pipelines on nemo atm
import time
import inspect
import functools
import logging

from tqdm import tqdm

FILE_FORMAT = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
CONSOLE_FORMAT = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


class DefaultLogger(logging.Logger):

    """ A conveniece class for logger initialization """

    def __init__(self, name, log_file='/tmp'):
        logging.Logger.__init__(self, name)

        # adding a file and a channel handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(FILE_FORMAT)
        sh = logging.StreamHandler()
        sh.setFormatter(CONSOLE_FORMAT)
        self.addHandler(fh)
        self.addHandler(sh)


def xileh_log_this(custom_logger=None, log_file='/tmp/xileh.log'):
    """ Wrapper to decorate a pipeline function

    Note: We are using this to return a decorator with parameters for
    custom_logger and log_dir

    Important: As we decorate with this 'decorator_factory' we need
    to actually call it to get a decorator, thus is

    @xileh_log_this()
    def some_foo():

    instead of just

    @xileh_log_this
    def some_foo()

    The call is necessary even if no parameters are specified!


    Parameters
    ----------
    custom_logger : logging.Logger
        A custom logger to use for logging
    log_file : str
        Path to store the logging info at - only available with custom
        logger

    Returns
    -------
        Function decorated with at logger

    """

    def decorator(func):
        """ decorating a function using parameters from outside scope
            for custom_logger and log_dir
        """

        # First validate, if the function accepts a logger,
        # else raise an KeyError as presumeable the decorator
        # was not set on purpose ... => no need to log if no
        # call to logger is made
        if custom_logger is None:
            loggerw = DefaultLogger(func.__name__, log_file=log_file)
        else:
            loggerw = custom_logger

        if 'logger' not in inspect.signature(func).parameters:
            raise KeyError(f"Function {func.__name__} does not contain"
                           " a logger parameter, should be wrapped for"
                           " logging. Please add an arg or kwarg"
                           " called 'logger' or provide a custom_logger"
                           " to the wrapper directly.")

        @functools.wraps(func)
        def wrapped_f(*args, **kwargs):
            # if logger was not porvided in kwargs, use the
            # default one
            if 'logger' not in kwargs.keys():
                kwargs['logger'] = loggerw

            return func(*args, **kwargs)

        return wrapped_f

    return decorator


class PlainLogger(object):

    """ A plain logger which will write to a file (and console optionally) """

    def __init__(self, logfile, console_print=False, use_tqdm=True):
        """
        Parameters
        ----------
        logfile : str or Path
            path to the desired output file
        console_print : bool
            whether or not to also print to the console
        use_tqdm : bool
            whether or not to use tqdm.write intead of print
        """
        self.logfile = logfile

        # Message header, can be any callable returning a str
        self.head = self.default_msg_head
        self.delim = ' - '

        if console_print:
            if use_tqdm:
                self.print = self.print_with_console_tqdm
            else:
                self.print = self.print_with_console

    def print_with_console_tqdm(self, msg):
        tqdm.write(msg)
        print(msg, file=self.get_lf())

    def print_with_console(self, msg):
        print(msg)
        print(msg, file=self.get_lf())

    def print(self, msg):
        print(msg, file=self.get_lf())

    def get_lf(self):
        """ Get the log file in write mode """
        return open(self.logfile, 'a')

    def default_msg_head(self):
        """ The default message head to be printed """
        return time.asctime()

    def info(self, msg):
        self.print(self.delim.join([self.head(), 'INFO', msg]))

    def debug(self, msg):
        self.print(self.delim.join([self.head(), 'DEBUG', msg]))

    def warning(self, msg):
        self.print(self.delim.join(
            [self.head(), 'WARNING', msg]))

    def error(self, msg):
        self.print(self.delim.join([self.head(), 'ERROR', msg]))
