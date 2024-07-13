#!/bin/bash
set -e
for i in {0..143}
do
    echo "Running train.py with index $i"
    python train.py --index $i
done
