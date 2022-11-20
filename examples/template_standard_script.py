#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 20220123
#
# A simple standard setup I use for scripts which might also be run via cli

from fire import Fire
from xileh import xPData, xPipeline


# ================================= Functions =================================
def add_data_entity(pdata, name='new_test', size=4):
    """ Add a simply data entity to the container """
    pdata.add([1, 2, 3] * size, name)

    # Note, every function used in a pipeline needs to return the pdata object
    # --> this is to make explicit, that the function modifies the pdata object
    return pdata


# ================================= Pipeline ==================================
pl = xPipeline('test_pipeline', log_eval=False)
pl.add_steps(
    ('add_to_data', add_data_entity),
    ('add_to_data_2', add_data_entity, {'name': 'another_test'}),
)


# =================================== CLI =====================================
def main(size=4):
    pdata = xPData([], name='test_container')
    pl.set_step_kwargs('add_to_data_2', size=size * 2)
    pl.eval(pdata)
    print(pdata)
    print(pdata['new_test'].data)
    print(pdata['another_test'].data)


if __name__ == '__main__':
    Fire(main)
