#!/usr/bin/env python3
"""
Data handling for imars3d.
"""
import param
import numpy as np
import dxchange
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, List, Callable
from tqdm.contrib.concurrent import process_map


# setup module level logger
logger = param.get_logger()
logger.name = __name__


# use _func to avoid sphinx pulling it into docs
def _load_images(
    filelist: List[str],
    desc: str,
    max_workers: int,
) -> np.ndarray:
    """
    Load image data via dxchange.
    
    Parameters
    ----------
    filelist:
        List of images filenames/path for loading via dxchange.
    desc:
        Description for progress bar.
    max_workers:
        Maximum number of processes allowed during loading.

    Returns
    -------
        Image array stack.
    """
    # figure out the file type and select corresponding reader from dxchange
    file_ext = Path(filelist[0]).suffix.lower()
    if file_ext in (".tif", ".tiff"):
        reader = dxchange.read_tiff
    elif file_ext == ".fits":
        reader = dxchange.read_fits
    else:
        logger.error(f"Unsupported file type: {file_ext}")
        raise ValueError("Unsupported file type.")
    # read the data into numpy array via map_process
    rst = process_map(
        partial(_forgiving_reader, reader=reader),
        filelist,
        max_workers=max_workers,
        desc=desc,
    )
    # return the results
    return np.array([me for me in rst if me is not None])


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
