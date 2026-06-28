"""
Wrapper script to run train_ann.py with a reduced number of Optuna trials
for fast end-to-end verification.
"""

import sys
import os

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODEL_DIR)

import train_ann

# Save original optimize method
original_optimize = train_ann.optuna.Study.optimize

def mock_optimize(self, func, n_trials=None, **kwargs):
    print(f"\n[MOCK] Overriding n_trials from {n_trials} to 3 for fast verification...\n", flush=True)
    return original_optimize(self, func, n_trials=3, **kwargs)

# Patch optuna.Study.optimize
train_ann.optuna.Study.optimize = mock_optimize

if __name__ == "__main__":
    train_ann.main()
