FROM carla-step4

USER root

RUN apt-get install -y python3.7 python3.7-dev 
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

RUN apt-get update -y
RUN apt-get install -y libjpeg8 libtiff5 # carla python package dependency
RUN apt-get install -y python3-pip
RUN apt-get install -y fontconfig


# Change the user from root back to carla, otherwise, carla will raise error "refusing to run with root privilege"
USER carla

# pip installation does not require root privilege
RUN pip3 install -r /home/carla/carla/PythonAPI/carla/requirements.txt
RUN pip3 install --upgrade pip # old pip3 fail to install matplotlib
RUN pip3 install ipython
RUN pip3 install matplotlib
RUN pip3 install torchaudio==0.10.1 
# For this error https://stackoverflow.com/questions/69968477/runtimeerror-cuda-error-no-kernel-image-is-available-for-execution-on-the-devi
RUN pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install opencv-python
RUN pip3 install tensorboardX tensorboard
RUN pip3 install ipykernel tqdm

