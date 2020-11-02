# -*- coding: utf-8 -*-
"""Unified logger.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import datetime as dt
import logging


PATH_LOGS = os.path.dirname(os.path.abspath(__file__))
PATH_LOGS = os.path.abspath(os.path.join(PATH_LOGS, '../logs'))
PATH_CUR_LOG = 'log_{0}.log'.format(
    dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
PATH_CUR_LOG = os.path.join(PATH_LOGS, PATH_CUR_LOG)


logger = logging.getLogger('main-log')
logger.setLevel(logging.INFO)

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
c_handler.setFormatter(logging.Formatter(
    '[%(levelname)s] %(asctime)s - %(message)s'))
logger.addHandler(c_handler)

f_handler = logging.FileHandler(PATH_CUR_LOG)
f_handler.setLevel(logging.INFO)
f_handler.setFormatter(logging.Formatter(
    '[%(asctime)s - %(name)s - %(levelname)s] %(message)s'))
logger.addHandler(f_handler)
