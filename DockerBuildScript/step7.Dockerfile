FROM carla-step6

USER carla
# RUN cd alone is useless! See https://stackoverflow.com/questions/58847410/difference-between-run-cd-and-workdir-in-dockerfile
RUN git clone https://github.com/tensorflow/models.git /home/carla/tensorflow_models
RUN git clone https://github.com/shangtse/robust-physical-attack.git /home/carla/robust_physical_attack

ARG SHA1=fe748d4
RUN cd /home/carla/tensorflow_models && git reset --hard $SHA1

# Install protoc
# Taken from https://stackoverflow.com/questions/47704968/protoc-command-not-found-linux
ARG PROTOC_VER=3.0.0
ARG PROTOC_ZIP=protoc-$PROTOC_VER-linux-x86_64.zip
RUN wget https://github.com/google/protobuf/releases/download/v$PROTOC_VER/$PROTOC_ZIP

USER root
RUN unzip $PROTOC_ZIP bin/protoc -d /usr/local
RUN unzip $PROTOC_ZIP include/* -d /usr/local
RUN chmod a+rx /usr/local/bin/protoc
RUN chmod -R a+rx /usr/local/include/google

USER carla
WORKDIR /home/carla/tensorflow_models/research



# Compile protoc.
RUN protoc object_detection/protos/*.proto --python_out=.

# Below packages taken from https://github.com/tensorflow/models/blob/fe748d4a4a1576b57c279014ac0ceb47344399c4/research/object_detection/g3doc/installation.md
# Install tensorflow object detection dependencies
RUN pip3 install  lucid==0.3.1 
RUN pip3 install  tensorflow-gpu==1.13.2
RUN pip3 install  Cython==0.29.13
RUN pip3 install  contextlib2
RUN pip3 install  pillow
RUN pip3 install  lxml
RUN pip3 install  jupyter
RUN pip3 install  protobuf==3.19 # downgrade protobuf
RUN pip3 install matplotlib prettytable
RUN pip3 install opencv-python

RUN patch -p1 -d . < /home/carla/robust_physical_attack/object_detection_api.diff
RUN LUCID_DIR=$(pip3 show lucid | grep "Location:" | cut -d" " -f2)/lucid && \
    patch -p1 -d $LUCID_DIR < /home/carla/robust_physical_attack/lucid.diff
ENV PYTHONPATH "$PYTHONPATH:/home/carla/tensorflow_models/research:/home/carla/tensorflow_models/research/slim"
WORKDIR /home/carla/carla

# Install NVIDIA cuda for tensorflow
# Taken from https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130
# Taken from https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation
USER root
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804-keyring.gpg 
RUN mv cuda-ubuntu1804-keyring.gpg /usr/share/keyrings/cuda-archive-keyring.gpg
# Add cuda-libraries repository, which installs libcublas, libcudart, libcusolver, libnvblas for tensorflow
RUN echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" | tee /etc/apt/sources.list.d/cuda-ubuntu1804-x86_64.list
# Add cudnn repository, which installs libcudnn for tensorflow
RUN echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/ /" | tee /etc/apt/sources.list.d/cudnn-ubuntu1804-x86_64.list
RUN apt-get update && apt-get install -y cuda-cublas-10-0 cuda-cudart-10-0 cuda-cusolver-10-0
RUN apt-get install -y libcudnn7=7.4.2.24-1+cuda10.0
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64"
ENV PATH="$PATH:/home/carla/.local/bin"
USER carla
