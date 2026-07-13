"""Reproducibility wrapper for the ML-1 experiment family (protocol item 5).

Runs Stage 0 (engine fidelity GATE 0) + Stage 1 (the pre-registered CORE 12
signal-matrix experiments). All machinery lives in ml_track/:

    /opt/anaconda3/bin/python3 exp_ml1_stage01.py
        == /opt/anaconda3/bin/python3 -m ml_track.run_all_stage1

Outputs: ml_track/results/ledger.csv, ml_track/results/STAGE1_REPORT.md,
results/v7/trials/ML_stage0.csv, results/v7/trials/ML_stage1.csv.
"""
import sys

sys.path.insert(0, ".")

from ml_track.run_all_stage1 import main

if __name__ == "__main__":
    main()
