#!/usr/bin/env python3
"""
Unit tests for backend data loading.
"""
import os
import pytest
import astropy.io.fits as fits
import dxchange
import numpy as np
from functools import partial
from unittest import mock
from pathlib import Path
from imars3d.backend.data import load_data
from imars3d.backend.data import _forgiving_reader
from imars3d.backend.data import _load_images
from imars3d.backend.data import _load_by_file_list
from imars3d.backend.data import _get_filelist_by_dir
from imars3d.backend.data import _extract_rotation_angles


@pytest.fixture(scope="module")
def test_data():
    # create some test data
    data = np.ones((3, 3))
    dxchange.write_tiff(data, "test")
    hdu = fits.PrimaryHDU(data)
    hdu.writeto('test.fits')
    yield
    # remove the test data
    os.remove("test.tiff")
    os.remove("test.fits")


@mock.patch("imars3d.backend.data._extract_rotation_angles", return_value=4)
@mock.patch("imars3d.backend.data._get_filelist_by_dir", return_value=("1", "2", "3"))
@mock.patch("imars3d.backend.data._load_by_file_list", return_value=(1, 2, 3))
def test_load_data(_load_by_file_list, _get_filelist_by_dir, _extract_rotation_angles):
    # error_0: incorrect input argument types
    with pytest.raises(ValueError):
        load_data(ct_files=1, ob_files=[], dc_files=[])
        load_data(ct_files=[], ob_files=[], dc_files=[], ct_fnmatch=1)
        load_data(ct_files=[], ob_files=[], dc_files=[], ob_fnmatch=1)
        load_data(ct_files=[], ob_files=[], dc_files=[], dc_fnmatch=1)
        load_data(ct_files=[], ob_files=[], dc_files=[], max_workers="x")
        load_data(ct_dir=1, ob_dir="/tmp", dc_dir="/tmp")
        load_data(ct_dir="/tmp", ob_dir=1, dc_dir="/tmp")
    # error_1: out of bounds value
    with pytest.raises(ValueError):
        load_data(ct_files=[], ob_files=[], dc_files=[], max_workers=-1)
    # error_2: mix usage of function signature 1 and 2
    with pytest.raises(ValueError):
        load_data(ct_files=[], ob_files=[], dc_files=[], ct_dir="/tmp", ob_dir="/tmp")
    # case_1: load data from file list
    rst = load_data(ct_files=["1", "2"], ob_files=["3", "4"], dc_files=["5", "6"])
    assert rst == (1, 2, 3, 4)
    # case_2: load data from given directory
    rst = load_data(ct_dir="/tmp", ob_dir="/tmp", dc_dir="/tmp")
    assert rst == (1, 2, 3, 4)


def test_forgiving_reader():
    # correct usage
    goodReader = lambda x: x
    assert _forgiving_reader(filename="test", reader=goodReader) == "test"
    # incorrect usage, but bypass the exception
    badReader = lambda x: x/0
    assert _forgiving_reader(filename="test", reader=badReader) is None


def test_load_images(test_data):
    func = partial(_load_images, desc="test", max_workers=2)
    # error case: unsupported file format
    incorrect_filelist = ["file1.bad", "file2.bad"]
    with pytest.raises(ValueError):
        rst = func(filelist=incorrect_filelist)
    # case1: tiff
    tiff_filelist = ["test.tiff", "test.tiff"]
    rst = func(filelist=tiff_filelist)
    assert rst.shape == (2, 3, 3)
    # case2: fits
    fits_filelist = ["test.fits", "test.fits"]
    rst = func(filelist=fits_filelist)
    assert rst.shape == (2, 3, 3)


@mock.patch("imars3d.backend.data._load_images", return_value="a")
def test_load_by_file_list(_load_images):
    # error_1: ct empty
    with pytest.raises(ValueError):
        _load_by_file_list(ct_files=[], ob_files=[])
    # error_2: ob empty
    with pytest.raises(ValueError):
        _load_by_file_list(ct_files=["dummy"], ob_files=[])
    # case_1: load all three
    rst = _load_by_file_list(ct_files=["a.tiff"], ob_files=["a.tiff"], dc_files=["a.tiff"])
    assert rst == ("a", "a", "a")
    # case_2: load only ct and ob
    rst = _load_by_file_list(ct_files=["a.tiff"], ob_files=["a.tiff"])
    assert rst == ("a", "a", None)


def test_get_filelist_by_dir(test_data):
    # error_1: ct_dir does not exists
    with pytest.raises(ValueError):
        _get_filelist_by_dir(ct_dir="dummy", ob_dir="/tmp", dc_dir="/tmp")
    # error_2: ob_dir does not exists
    with pytest.raises(ValueError):
        _get_filelist_by_dir(ct_dir="/tmp", ob_dir="dummy", dc_dir="/tmp")
    # case_1: load all three
    cwd = str(Path.cwd())
    rst = _get_filelist_by_dir(ct_dir=cwd, ob_dir=cwd, dc_dir=cwd)
    # case_2: 
    # case_2: load ct, and detect ob and df from metadata
    # case_3: 
    
    pass


def test_extract_rotation_angles():
    pass


if __name__ == "__main__":
    pytest.main([__file__])
