#!/usr/bin/env python3
"""
Data handling for imars3d.
"""
import re
import param
import multiprocessing
import numpy as np
import dxchange
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, List, Callable
from tqdm.contrib.concurrent import process_map


# setup module level logger
logger = param.get_logger()
logger.name = __name__


class load_data(param.ParameterizedFunction):
    """
    Load data with given input
    
    Parameters
    ---------
    ct_files: str
        explicit list of radiographs
    ob_files: str
        explicit list of open beams
    dc_files: Optional[str]
        explicit list of dark current
    ct_dir: str
        directory contains radiographs
    ob_dir: str
        directory contains open beams
    dc_dir: Optional[str]
        directory contains dark currents
    ct_regex: Optional[str]
        regular expression for down selecting radiographs
    ob_regex: Optional[str]
        regular expression for down selecting open beams
    dc_regex: Optional[str]
        regular expression for down selecting dark current
    max_workers: Optional[int]
        maximum number of processes allowed during loading, default to use as many as possible.

    Returns
    -------
        radiograph stacks, obs, dcs and omegas as numpy.ndarray

    Notes
    -----
        There are two main signatures to load the data:
        1. load_data(ct_files=ctfs, ob_files=obfs, dc_files=dcfs)
        2. load_data(ct_dir=ctdir, ob_dir=obdir, dc_dir=dcdir)

        The two signatures are mutually exclusive, and dc_files and dc_dir are optional
        in both cases as some experiments do not have dark current measurements.

        The regex selectors are applicable in both signature, which help to down-select
        files if needed. Default is set to "*", which selects everything.
        Also, if ob_regex and dc_regex are set to "None" in the second signature call, the
        data loader will attempt to read the metadata embedded in the ct file to find obs
        and dcs with similar metadata.

        Currently, we are using a forgiving reader to load the image where a corrupted file
        will not block reading other data.
    """
    #
    ct_files = param.List(doc="list of all ct files to load")
    ob_files = param.List(doc="list of all ob files to load")
    dc_files = param.List(doc="list of all dc files to load")
    #
    ct_dir = param.Foldername(doc="radiograph directory")
    ob_dir = param.Foldername(doc="open beam directory")
    dc_dir = param.Foldername(doc="dark current directory")
    # NOTE: we need to provide a default value here as param.String default to "", which will
    #       not trigger dict.get(key, value) to get the value as "" is not None.
    ct_regex = param.String(default="\b*", doc="regex for selecting ct files from ct_dir")
    ob_regex = param.String(default="\b*", doc="regex for selecting ob files from ob_dir")
    dc_regex = param.String(default="\b*", doc="regex for selecting dc files from dc_dir")
    # NOTE: 0 means use as many as possible
    max_workers = param.Integer(default=0, bounds=(0, None), doc="Maximum number of processes allowed during loading")

    def __call__(self, **params):
        """
        This makes the class behaves like a function.
        """
        # type*bounds check via Parameter
        _ = self.instance(**params)
        # sanitize arguments
        params = param.ParamOverrides(self, params)
        # type validation is done, now replacing max_worker with an actual integer
        self.max_workers = multiprocessing.cpu_count() - 2 if params.max_workers == 0 else params.max_workers
        logger.debug(f"max_worker={self.max_workers}")

        # multiple dispatch
        # NOTE:
        #    use set to simplify call signature checking
        sigs = set([k.split("_")[-1] for k in params.keys() if "regex" not in k])
        if sigs == {"files", "dir"}:
            logger.error("Files and dir cannot be used at the same time")
            raise ValueError("Mix usage of allowed signature.")
        elif sigs == {"files"}:
            logger.debug("Load by file list")
            ct, ob, dc = _load_by_file_list(
                    ct_files=params.get("ct_files"),
                    ob_files=params.get("ob_files"),
                    dc_files=params.get("dc_files", []),  # it is okay to skip dc
                    ct_regex=params.get("ct_regex", "\b*"),  # incase None got leaked here
                    ob_regex=params.get("ob_regex", "\b*"),
                    dc_regex=params.get("dc_regex", "\b*"),
                )
            ct_files=params.get("ct_files")
        elif sigs == {"dir"}:
            logger.debug("Load by directory")
            ct, ob, dc = _load_by_dir(
                    ct_dir=params.get("ct_dir"),
                    ob_dir=params.get("ob_dir"),
                    dc_dir=params.get("dc_dir", []),  # it is okay to skip dc
                    ct_regex=params.get("ct_regex", "\b*"),  # incase None got leaked here
                    ob_regex=params.get("ob_regex", "\b*"),
                    dc_regex=params.get("dc_regex", "\b*"),
                )
            ct_files = list(Path(params.get("ct_dir")).glob(params.get("ct_regex", "\b*")))
        else:
            logger.warning("Found unknown input arguments, ignoring.")

        # extracting omegas from
        # 1. filename
        # 2. metadata (only possible for Tiff)
        rot_angles = _extract_rotation_angles(ct_files)

        # return everything
        return ct, ob, dc, rot_angles


def _extract_rotation_angles(filelist: List[str]) -> np.ndarray:
    """
    """
    raise NotImplementedError


def _load_by_dir(
        ct_dir: str,
        ob_dir: str,
        dc_dir: Optional[str],
        ct_regex: str,
        ob_regex: str,
        dc_regex: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data by directory
    """
    raise NotImplementedError
    # get all files
    ct_files = list(Path(ct_dir).glob(ct_regex))
    ob_files = list(Path(ob_dir).glob(ob_regex))
    dc_files = list(Path(dc_dir).glob(dc_regex)) if dc_dir not in  ([], None) else []

    # load data
    return _load_by_file_list(ct_files, ob_files, dc_files, ct_regex, ob_regex, dc_regex)


# use _func to avoid sphinx pulling it into docs
def _load_by_file_list(
    ct_files: List[str],
    ob_files: List[str],
    dc_files: Optional[List[str]],
    ct_regex: Optional[str],
    ob_regex: Optional[str],
    dc_regex: Optional[str],
    max_workers: int = 0,
) -> Tuple[np.ndarray]:
    """
    Use provided list of files to load images into memory.
    """
    # empty list is not allowed
    if ct_files == []:
        logger.error("ct_files is [].")
        raise ValueError("ct_files cannot be empty list.")
    if ob_files == []:
        logger.error("ob_files is [].")
        raise ValueError("ob_files cannot be empty list.")
    if dc_files == []:
        logger.warning("dc_files is [].")

    # explicit list is the most straight forward solution
    # -- radiograph
    re_ct = re.compile(ct_regex)
    ct = _load_images(
        filelist=[ctf for ctf in ct_files if re_ct.match(ctf)],
        desc="ct",
        max_workers=max_workers,
    )
    # -- open beam
    re_ob = re.compile(ob_regex)
    ob = _load_images(
        filelist=[obf for obf in ob_files if re_ob.match(obf)],
        desc="ob",
        max_workers=max_workers,
    )
    # -- dark current
    if dc_files == []:
        dc = None
    else:
        re_dc = re.compile(dc_regex)
        dc = _load_images(
            filelist=[dcf for dcf in dc_files if re_dc.match(dcf)],
            desc="dc",
            max_workers=max_workers,
        )
    #
    return ct, ob, dc


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
    reader:
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
