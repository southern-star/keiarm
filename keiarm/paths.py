"""Filesystem locations shared across the project."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SO101_DIR = REPO_ROOT / "third_party" / "robotstudio_so101"
SO101_XML = SO101_DIR / "so101.xml"
