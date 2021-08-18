#!/bin/bash
#docker run -it --rm -p 8888:8888 \
nvidia-docker run --runtime=nvidia -it --rm -p 8888:8888 \
  -v $(pwd):/code \
  mkuchnik/pcr \
  "/bin/bash"
