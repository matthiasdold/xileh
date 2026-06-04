#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 2021-03-26

import pytest
import inspect

from logging import Logger
from xileh.utils.logger import xileh_log_this, DefaultLogger


@pytest.fixture
def xileh_logged_func():

    @xileh_log_this()
    def foo(logger=Logger('default')):
        print("bar")

    return foo


def test_error_if_no_logger_in_parameters():
    """ Should raise a key error """
    with pytest.raises(KeyError):
        @xileh_log_this()
        def foo_without_logger():
            return 0


def test_is_wrapped(xileh_logged_func):
    assert '__wrapped__' in xileh_logged_func.__dict__.keys()


def test_default_decorator(xileh_logged_func):
    """ Decorator should create a logger with function name """
    sig = inspect.signature(xileh_logged_func)

    # it is there
    assert sig.parameters['logger'].name == 'logger'

    # the default named one was created
    assert sig.parameters['logger'].default.name == 'default'


def test_provide_custom_logging():
    @xileh_log_this(custom_logger=Logger('custom_logger'))
    def foo_with_custom_logger(logger=Logger('default')):
        return logger.name

    assert foo_with_custom_logger() == 'custom_logger'

    # custom file only without custom logger -> else assume already set
    # in custom logger
    @xileh_log_this(log_file='/tmp/testfile.log')
    def foo_with_custom_log_dir_wo_logger(logger=Logger('default')):
        return logger

    logger = foo_with_custom_log_dir_wo_logger()
    assert isinstance(logger, DefaultLogger)
    assert logger.handlers[0].stream.name == '/tmp/testfile.log'

    @xileh_log_this(custom_logger=Logger('custom_logger'),
                    log_file='/tmp/custom/file.log')
    def foo_with_custom_log_dir_w_logger(logger=Logger('default')):
        return logger.name

    assert foo_with_custom_log_dir_w_logger() == 'custom_logger'
