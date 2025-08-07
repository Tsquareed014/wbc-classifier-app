# run_tests.py
import sys
import os
import pytest


# Run pytest with the specified arguments
pytest.main(["--maxfail=1", "--disable-warnings", "-q", "test_model_loader.py"])
