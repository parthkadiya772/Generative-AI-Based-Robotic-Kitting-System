FROM nvcr.io/nvidia/isaac-sim:6.0.1

USER root

# 1. Install build tools, GCC-12, CUDA toolkit, and Vulkan dev libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    gcc-12 \
    g++-12 \
    nvidia-cuda-toolkit \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libbenchmark-dev \
    libgflags-dev \
    libvulkan-dev \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Set GCC-12 as default host compiler
ENV CC=/usr/bin/gcc-12
ENV CXX=/usr/bin/g++-12
ENV CUDAHOSTCXX=/usr/bin/g++-12
ENV CUDACXX=/usr/bin/nvcc
ENV CUDA_HOME=/usr
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

# 2. Install PyTorch + packaging into Isaac Sim's Python runtime
RUN /isaac-sim/python.sh -m pip install --upgrade pip && \
    /isaac-sim/python.sh -m pip install packaging && \
    /isaac-sim/python.sh -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Overwrite system CUPTI library with CUDA 12.4 version to prevent linker collision
RUN cp -f /isaac-sim/kit/python/lib/python3.12/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12* /usr/lib/x86_64-linux-gnu/ && \
    ldconfig

# 4. Build and install core nvblox library
WORKDIR /root
RUN git clone https://github.com/nvidia-isaac/nvblox.git && \
    cd nvblox && \
    mkdir build && cd build && \
    TORCH_PATH=$(/isaac-sim/python.sh -c 'import torch.utils; print(torch.utils.cmake_prefix_path)') && \
    cmake .. \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" \
    -DNVBLOX_CUDA_ARCH="75;80;86;89" \
    -DCMAKE_PREFIX_PATH="${TORCH_PATH}" \
    -DBUILD_RENDERER=OFF \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release && \
    make -j4 && \
    make install && \
    ldconfig

# 5. Install nvblox_torch bindings
RUN cd /root/nvblox/nvblox_torch && \
    /isaac-sim/python.sh -m pip install -e . --no-build-isolation

# 6. Install cuRobo into Isaac Sim Python runtime
RUN git clone https://github.com/NVlabs/curobo.git && \
    cd curobo && \
    /isaac-sim/python.sh -m pip install -e . --no-build-isolation

WORKDIR /isaac-sim