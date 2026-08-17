#!/usr/bin/env python3
"""
run_all.py — one-command runner for the whole pipeline.

Executes German_Electricity_Uncertainity.ipynb from top to bottom, in order,
so every intermediate file (data_part2.pkl, results_part8.npy, ...) is generated
automatically. A fresh clone needs only the raw CSV in data/ — this script
produces everything else.

Usage:
    pip install -r requirements.txt
    python run_all.py

The executed notebook (with outputs) is written to:
    German_Electricity_Uncertainity.executed.ipynb
"""
import sys, subprocess, os

NOTEBOOK = "German_Electricity_Uncertainity.ipynb"
OUTPUT   = "German_Electricity_Uncertainity.executed.ipynb"

def main():
    if not os.path.exists(NOTEBOOK):
        sys.exit(f"ERROR: {NOTEBOOK} not found. Run this from the repo root.")

    # Ensure jupyter's nbconvert is available
    try:
        import nbconvert  # noqa
    except ImportError:
        print("Installing jupyter/nbconvert ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "jupyter", "nbconvert", "nbformat", "ipykernel"])

    print(f"Executing {NOTEBOOK} top-to-bottom (this trains all models; "
          f"it can take a while)...")
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute",
        "--ExecutePreprocessor.timeout=-1",   # no per-cell timeout
        "--output", OUTPUT,
        NOTEBOOK,
    ]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"\n✓ Done. Executed notebook saved to {OUTPUT}")
        print("  All intermediate .pkl / .npy / .json files are now generated.")
    else:
        sys.exit("\nExecution failed. Open the notebook and run cells manually "
                 "to see the error, or check that the data CSV is in data/.")

if __name__ == "__main__":
    main()
