#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 20220201
#
# Generic tests for saving and loading data

import pandas as pd
import numpy as np


df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
s = pd.Series({'my_s': [1, 2, 3, 4]})
a = np.ones((3, 3, 6))
