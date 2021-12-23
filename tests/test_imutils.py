import pytest

import imutils

def test_imutils():
    assert imutils.__version__ == "0.0.1"


if __name__ == "__main__":
    pytest.main([__file__])
