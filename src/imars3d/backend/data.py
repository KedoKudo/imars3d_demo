#!/usr/bin/env python3
"""
Data handling for imars3d.
"""
import param
import numpy as np
from functools import partial
from typing import Optional, Tuple, List, Callable
from tqdm.contrib.concurrent import process_map


# setup module level logger
logger = param.get_logger()
logger.name = __name__


# use _func to avoid sphinx pulling it into docs
def _forgiving_reader(
    filename: str,
    reader: Optional[Callable],
) -> Optional[np.ndarray]:
    """
    Skip corrupted file, but inform the user about the issue.
    
    Parameters
    ----------
    filename:
        input filename
    rader:
        callable reader function that consumes the filename

    Returns
    -------
        image as numpy array
    """
    try:
        return reader(filename)
    except:
        logger.error(f"Cannot read {filename}, skipping.")
        return None