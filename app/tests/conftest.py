import os
import sys

# Make the application package importable (utils.*, config) when running pytest
# from the project. The app itself is launched from this same directory.
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
