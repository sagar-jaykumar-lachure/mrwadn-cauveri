# Development Notes

## Import Paths

The scripts `train.py`, `predict.py`, `api.py`, and `demo.py` use `sys.path.append()` to add the project root to the Python path. This is intentional for ease of use:

- Allows running scripts directly without installing the package
- Works from any directory
- Simplifies development and testing

For production deployment, consider:
- Installing as a proper Python package using `setup.py`
- Using Docker containers with proper PYTHONPATH
- Setting PYTHONPATH environment variable

## Pandas Methods

The codebase uses modern pandas methods:
- `df.ffill()` instead of deprecated `df.fillna(method='ffill')`
- `df.bfill()` instead of deprecated `df.fillna(method='bfill')`

These are compatible with pandas 2.0+ and future versions.

## Configuration

Both LSTM and GRU models share the same `lstm` configuration section for parameters like `sequence_length`. This is intentional as they have identical hyperparameters for this use case.
