# step 1 - Install X Server
# Build with 
# docker build -t carla-step1 -f step1.Dockerfile .
# Run with
# docker run -it --rm --privileged --runtime nvidia --net=host -e DISPLAY=$DISPLAY carla-step1 bash


FROM carla-step0

USER root

RUN packages='libsdl2-2.0 xserver-xorg libvulkan1 libomp5' && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y $packages --no-install-recommends

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update ; \
    apt-get install -y wget software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|apt-key add - && \
    apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-8 main" && \
    apt-get update ; \
    apt-get install -y build-essential \
  		       clang-8 \
			   lld-8 \
  		       g++-7 \
  		       cmake \
  		       ninja-build \
  		       libvulkan1 \
  		       python \
  		       python-pip \
  		       python-dev \
  		       python3-dev \
  		       python3-pip \
  		       libpng-dev \
  		       libtiff5-dev \
  		       libjpeg-dev \
  		       tzdata \
  		       sed \
  		       curl \
  		       unzip \
  		       autoconf \
  		       libtool \
  		       rsync \
  		       libxml2-dev \
  		       git \
  		       aria2 && \
    pip3 install -Iv setuptools==47.3.1 && \
    pip3 install distro && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/lib/llvm-8/bin/clang++ 180 && \
    update-alternatives --install /usr/bin/clang clang /usr/lib/llvm-8/bin/clang 180

	# To resolve this issue https://github.com/carla-simulator/carla/issues/420#issuecomment-2275252760
RUN apt install -y mono-runtime 

USER carla
