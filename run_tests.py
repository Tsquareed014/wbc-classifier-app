# run_tests.py
import sys
import os
import pytest

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Run pytest with the specified arguments
pytest.main(["--maxfail=1", "--disable-warnings", "-q", "tests/"])
