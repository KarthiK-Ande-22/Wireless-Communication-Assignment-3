#!/usr/bin/env bash
# Reproduce every figure and number of the report.
# Run from the project root:   bash run_all.sh
set -e
export PYTHONDONTWRITEBYTECODE=1           # no __pycache__ folders

python3 Task-1/cp_design.py                  # Task 1: CP design
python3 Task-2/transmitter.py                # Task 2: transmitter check
python3 Task-3/channel.py                    # Task 3: channel realizations and check
python3 Task-4/receiver.py                   # Task 4: receiver check (no noise)
python3 Task-5/main.py --scenario 1          # Task 5 (a)-(c), Scenario 1, Ncp = 4
python3 Task-5/main.py --scenario 2          # Task 5 (a)-(c), Scenario 2, Ncp = 75
python3 Task-5/main.py --scenario 2 --cp 4   # Task 5 (d), Scenario 2, Ncp = 4
python3 Task-5/cp_sweep.py                   # extra: every Ncp from 0 to 80
