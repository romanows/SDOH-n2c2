#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Will pass along flags from the command-line to the unittest discover command.

STATUS=

# Explicitly avoid running on non-tests directories/modules.  If "unittest
# discover" runs on other directories, it seems to import it, which can lead to
# long delays for modules with long import times.  For example, anything that
# needs to import Tensorflow.
for d in tests; do
  if [[ -d "$d" ]]; then
    echo
    echo "##########################################"
    echo "# Discovering tests in '$d'"
    echo "##########################################"
    echo
    python -m unittest discover "$@" -t . -s "$d" || STATUS=$?
  fi
done

if [[ -z "$STATUS" ]]; then
  echo "No tests"
elif [[ $STATUS == 0 ]]; then
  echo "PASS"
else
  echo "FAIL"
  exit $STATUS
fi
