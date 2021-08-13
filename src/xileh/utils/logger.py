#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 2021-03-25
#
# Definition of the pipeline logger
#
# Target for usage: Apply the logger as a decorator

import inspect
import functools
import logging

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
