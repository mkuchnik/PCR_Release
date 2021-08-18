#!/bin/bash
set -e

python3 -m pip install -r requirements.txt
python3 -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/ nvidia-dali-cuda100
