#!/usr/bin/bash

set -eufx -o pipefail

bsub -q bora python3 invert.py -r TV -n 50000 --lamb 0.01 -i
sleep 5
bsub -q bora python3 invert.py -r TV -n 50000 --lamb 0.001 -i
sleep 5
bsub -q bora python3 invert.py -r TV -n 50000 --lamb 0.0001 -i
sleep 5
bsub -q bora python3 invert.py -r TV -n 50000 --lamb 0.00001 -i
sleep 5
bsub -q bora python3 invert.py -r TV -n 50000 --lamb 0.000001 -i
sleep 5
bsub -q bora python3 invert.py -r TV -n 50000 --lamb 0.0000001 -i
