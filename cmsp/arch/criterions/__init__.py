import sys
import os

sys.path.append(os.path.dirname(sys.path[0]))
import importlib

# automatically import any Python files in the criterions/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("cmsp.arch.criterions." + file_name)
