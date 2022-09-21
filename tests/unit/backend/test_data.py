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
from imars3d.backend.data import _forgiving_reader
from imars3d.backend.data import _load_images


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


def test_forgiving_reader():
    # correct usage
    goodReader = lambda x: x
    assert _forgiving_reader(filename="test", reader=goodReader) == "test"
    # incorrect usage, but bypass the exception
    badReader = lambda x: x/0
    assert _forgiving_reader(filename="test", reader=badReader) is None


if __name__ == "__main__":
    pytest.main([__file__])
