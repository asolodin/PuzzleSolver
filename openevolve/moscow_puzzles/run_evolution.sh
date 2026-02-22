#!/bin/bash
# Wrapper script to run OpenEvolve with the correct dataset

if [ $# -lt 1 ]; then
    echo "Usage: $0 <problem_file> [additional_args...]"
    echo "Example: $0 problem_250.json --iterations 20"
    exit 1
fi

export PROBLEM_FILE=$1
shift  # Remove first argument

# Run OpenEvolve
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml "$@"
