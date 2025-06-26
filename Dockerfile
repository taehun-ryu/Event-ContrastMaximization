# Base: Ubuntu 22.04 with build essentials
FROM ubuntu:22.04

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# 1. Basic dependencies
RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    libeigen3-dev \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgl1-mesa-dev \
    libglew-dev \
    libxi-dev \
    libx11-dev \
    libxrandr-dev \
    libxxf86vm-dev \
    libxinerama-dev \
    libusb-1.0-0-dev \
    libopenexr-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libdcmtk-dev \
    libboost-all-dev \
    libgtk-3-dev \
    libglfw3-dev \
    libopencv-dev \
    python3 \
    python3-pip \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libyaml-cpp-dev \
    && apt clean && rm -rf /var/lib/apt/lists/*

# 2. Install ceres-solver (v2.2.0)
WORKDIR /opt
RUN wget http://ceres-solver.org/ceres-solver-2.2.0.tar.gz \
 && tar zxf ceres-solver-2.2.0.tar.gz \
 && rm ceres-solver-2.2.0.tar.gz \
 && cd ceres-solver-2.2.0 \
 && mkdir build && cd build \
 && cmake .. \
 && make -j$(nproc) && make install \
 && ldconfig

# 3. Install Pangolin (0.9.3)
WORKDIR /opt
RUN git clone https://github.com/stevenlovegrove/Pangolin.git --branch v0.9.3 --depth 1 \
 && mkdir -p Pangolin/build && cd Pangolin/build \
 && cmake .. -DBUILD_PANGOLIN_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release \
 && make -j$(nproc) && make install \
 && ldconfig

# 4. Set working directory to your IBEC3 project
WORKDIR /IBEC3

