"""Test session guard (review round 4): every core.* module resolves
DATA_DIR at import time, and a migration that runs un-mocked inside a test
would os.replace() $DATA_DIR/model_config.json. Force a fresh temp dir
BEFORE any test module imports core, whatever the shell exported."""
import os
import tempfile

os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="combo_test_session_")
