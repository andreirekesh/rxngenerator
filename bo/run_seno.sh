#!/bin/bash

# Activate the Conda environment
source /root/miniconda3/bin/activate frag_bench

# Run the Python script with command-line arguments
python seno.py "$@"