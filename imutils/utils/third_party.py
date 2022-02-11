"""
imutils/utils/third_party.py

utility script for handling messy import steps for 3rd party dependencies that may have predictable issues.
Initially motivated by the pip library `torchdata` changing their name to `torchdatasets` in response to the main torch framework appropriating the `torchdata` namespace.

Created on: Thursday Jan 27th, 2022
Created by: Jacob A Rose

"""

__all__ = ["torchdata"]

try:
    import torchdatasets as torchdata
except ModuleNotFoundError:
    try:
        import torchdata
    except:
        from types import SimpleNamespace
        torchdata = SimpleNamespace()
