#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Runs extract-pretraining.py in the context of a Slurm array job.  Assumes the
# CWD is the project root directory.
#
# First arg should be the path to the NOTEEVENTS.csv.gz file or to the directory
# containing *.txt files.
#
# Second arg should be the path to the eventual output file.  The splitting
# process will cause many files with the ".split0001_1000" suffix to be written to
# the destination directory.  User is responsible for joining them later.

/usr/bin/time -v ./scripts/extract-pretraining.py --debug --split "$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT" "$1" "$2"
