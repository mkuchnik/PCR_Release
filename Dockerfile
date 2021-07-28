# Base image must at least have pytorch and CUDA installed.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:19.07-py3
FROM $BASE_IMAGE
ARG BASE_IMAGE
RUN echo "Installing Apex on top of ${BASE_IMAGE}"
# make sure we don't overwrite some existing directory called "apex"
WORKDIR /tmp/unique_for_apex
# uninstall Apex if present, twice to make absolutely sure :)
RUN pip uninstall -y apex || :
RUN pip uninstall -y apex || :
# SHA is something the user can touch to force recreation of this Docker layer,
# and therefore force cloning of the latest version of Apex
RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
WORKDIR /tmp/unique_for_apex/apex
RUN git checkout 50338df6280fd47832039725ec5bdcc202591222
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./



RUN apt-get update
RUN apt-get install -y cmake autoconf automake libtool curl make g++ unzip
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protobuf-all-3.6.1.tar.gz
RUN tar -xf protobuf-all-3.6.1.tar.gz
RUN cd protobuf-3.6.1 && ./configure && make && make check && make install
RUN ldconfig

RUN apt-get install -y python3-dev python3-pip
COPY install_python_deps.sh /
COPY requirements.txt /
RUN cd / && ./install_python_deps.sh
WORKDIR /code
