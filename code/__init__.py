"""Independent RF imaging and physics-based normal estimation."""

import importlib.util
import sys
import sysconfig
from pathlib import Path

# Keep debugger/console imports working alongside this package's requested name.
_spec = importlib.util.spec_from_file_location(
    "_radar_stdlib_code", Path(sysconfig.get_path("stdlib")) / "code.py"
)
if _spec is None or _spec.loader is None:
    raise ImportError("Cannot locate the Python standard library code module")
_console = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _console
_spec.loader.exec_module(_console)

InteractiveInterpreter = _console.InteractiveInterpreter
InteractiveConsole = _console.InteractiveConsole
interact = _console.interact
compile_command = _console.compile_command
__all__ = [
    "InteractiveConsole",
    "InteractiveInterpreter",
    "compile_command",
    "interact",
]
