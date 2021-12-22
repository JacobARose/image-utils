import pytest

import pyboilerplate


def test_pyboilerplate():
    assert pyboilerplate.__version__ == '0.0.1'


if __name__ == "__main__":
    pytest.main([__file__])
