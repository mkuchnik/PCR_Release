#!/bin/bash

cd proto
protoc -I=. --python_out=. *.proto
cd ..

mkdir build
cd build
gcc ../src/scan_only_jsk.c -o scan_only_jsk
