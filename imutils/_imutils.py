def imutils() -> str:
    """Return a string.

    Returns
    -------
    str
        A string

    Examples
    --------
    >>> imutils()
    'imutils'
    """
    return "imutils"

from pathlib import Path

ASSETS_DIR = Path(__file__).parent.parent / "assets"
SAMPLE_IMAGE_PATHS = [img_path for img_path in ASSETS_DIR.iterdir() if not str(img_path).endswith(".ipynb_checkpoints")]

print(f"ASSETS_DIR: {ASSETS_DIR}")
print(f"SAMPLE_IMAGE_PATHS: {SAMPLE_IMAGE_PATHS}")
