Development PC preparation
--------------------------

The following steps will walk you through the preparation for getting the project running on your local development PC.

### OS installation.

The PC should be running Ubuntu 22.04 OS. You may skip this step if the requirement is satisfied. 

### Graphic card driver installation

The development of the project was based on the NVIDIA 525 driver. You may use a higher-version driver that is backward compatible with this version.

To install graphic driver, go to All Apps \> Software & Update \> Additional Drivers. Select Using Nvidia driver metapackage from nvidia-driver-525 (proprietary) and click Apply Changes.

After installation, reboot the system to take effect. Check that your driver is installed correctly by opening a Terminal (Ctrl + Alt + T), and type

```
nvidia-smi
```

### Docker installation

Open a terminal, use the following command to install docker.

```
sudo apt install docker.io
```

After installation, add your current user to docker group using the following command, and then reboot your system to take effect.

```
sudo usermod -aG docker ${USER}
```
To check that docker has been properly installed, in a Terminal, and type

```
docker info
```

### NVIDIA docker installation

NVIDIA Docker, also known as NVIDIA Container Toolkit, is a set of tools and enhancements that enable GPU acceleration for Docker containers. This toolkit is necessary for the project. To install, use the following commands in a terminal.

```
sudo apt install curl

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g'| sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt install -y nvidia-container-toolkit
```

It may not take effect until you reboot. Reboot to verify that NIVIDA docker has been successfully installed, use the following command, and observe the output. If there is output like below, NVIDIA docker has been properly installed.

```
nvidia-container-toolkit --version
```

Lastly, configure NVIDIA docker with the following commands to complete the installation.

```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
