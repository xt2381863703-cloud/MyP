"""Stable repository paths shared by relocated runtime modules."""

from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
CONFIG_ROOT = CODE_ROOT / "config"
