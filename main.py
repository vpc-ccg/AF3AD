"""Shortcut entry point for the PO3AD root launcher."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "scripts" / "train_po3ad_integration.py"),
        run_name="__main__",
    )
