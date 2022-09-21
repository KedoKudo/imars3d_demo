#!/usr/bin/env python3
"""
Unit tests for backend data loading.
"""
import pytest
from unittest import mock
import numpy as np
from imars3d.backend.data import _forgiving_reader


def test_forgiving_reader():
    # correct usage
    goodReader = lambda x: x
    assert _forgiving_reader(filename="test", reader=goodReader) == "test"
    # incorrect usage, but bypass the exception
    badReader = lambda x: x/0
    assert _forgiving_reader(filename="test", reader=badReader) is None


if __name__ == "__main__":
    pytest.main([__file__])
