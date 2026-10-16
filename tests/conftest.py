"""Test session guard (review rounds 4-5): every core.* module resolves
DATA_DIR at import time, and a migration that runs un-mocked inside a test
would os.replace() $DATA_DIR/model_config.json. Force a fresh temp dir
BEFORE any test module imports core, whatever the shell exported; make the
repo importable under bare `pytest`; remove the dir at exit."""
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_SESSION_DIR = tempfile.mkdtemp(prefix="combo_test_session_")
os.environ["DATA_DIR"] = _SESSION_DIR
os.environ["AUCTION_FETCH_TODAY_OPENS"] = "0"   # never hit yfinance from a test
atexit.register(shutil.rmtree, _SESSION_DIR, ignore_errors=True)
