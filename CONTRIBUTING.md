# How to contribute

## Build and Test

```bash
git clone ...
cd ...
pip install -e .[develop]
pytest --doctest-modules
```

## Automatic check

Please make sure to run the following commands before contributing some code.
They install git hook scripts to avoid improper commits or pushes.

```bash
git clone ...
cd ...
pip install -e .[develop]
pre-commit install
pre-commit install -t pre-push
```
