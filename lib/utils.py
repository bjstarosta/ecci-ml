# -*- coding: utf-8 -*-
"""Utility functions.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import datetime


def generate_seed():
    """Return a semi-unique number for seeding a random state.

    Returns:
        int

    """
    return int(datetime.datetime.utcnow().strftime('%Y%m%d'))


def setup_path(path):
    """Check if passed directory exists, create it if it doesn't.

    Args:
        path (str): Path to directory.

    """
    if not os.path.isdir(path):
        os.mkdir(path)
