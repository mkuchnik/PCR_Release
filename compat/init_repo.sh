#!/bin/bash
set -e

git clone https://github.com/eigenteam/eigen-git-mirror.git | echo 0
git clone https://github.com/pybind/pybind11.git | echo 0

mkdir -p build
cd build
cmake ..
make -j 8

echo "Initialization is done"