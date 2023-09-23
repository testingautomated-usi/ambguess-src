FROM tensorflow/tensorflow:2.7.1-gpu

# Install dependencies
RUN /usr/bin/python3 -m pip install --upgrade pip
COPY requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt
RUN rm ./requirements.txt

# Copy license
COPY LICENSE ./LICENSE-AMBIGUESS

# Copy source files
COPY ./ambiguess ./ambiguess
COPY ./resources ./resources
RUN cd ./ambiguess


RUN \
    # Update nvidia GPG key
    rm /etc/apt/sources.list.d/cuda.list && \
    rm /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-key del 7fa2af80 && \
    apt-get update && apt-get install -y --no-install-recommends wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    # Update apt-get \
    apt-get -y update && \
    # Install GIT (required by foolbox)
    apt-get install -y git --fix-missing