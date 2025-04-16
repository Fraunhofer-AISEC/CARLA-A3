FROM carla-step7

USER root
# Install tkinter backend for matplotlib plotting
RUN apt-get -y install python3-tk
RUN apt-get -y install python3-setuptools


# Install Carla 0.9.15 PythonAPI and other packages
RUN  pip3 install file:///home/carla/carla/PythonAPI/carla/dist/carla-0.9.15-cp37-cp37m-linux_x86_64.whl
RUN pip3 install numpy==1.18.4
RUN wget https://files.pythonhosted.org/packages/18/b3/0bf5afdcf6ef95d2a343cd7865585a6efe5e3e727c1a4f3385c9935248cf/pygame-1.9.6-cp37-cp37m-manylinux1_x86_64.whl
RUN pip3 install pygame-1.9.6-cp37-cp37m-manylinux1_x86_64.whl
RUN python3 -m pip install --upgrade pip
RUN wget https://files.pythonhosted.org/packages/af/f3/683bf2547a3eaeec15b39cef86f61e921b3b187f250fcd2b5c5fb4386369/pandas-1.0.5-cp37-cp37m-manylinux1_x86_64.whl
RUN pip3 install pandas-1.0.5-cp37-cp37m-manylinux1_x86_64.whl



USER carla
