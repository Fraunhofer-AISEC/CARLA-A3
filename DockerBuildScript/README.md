


Development PC preparation
--------------------------

The following steps will walk you through the preparation for getting the project running on your local development PC.

### OS installation.

The PC should be running Ubuntu 22.04 OS. You may skip this step if the requirement is satisfied. Else, follow [^1] for the installation steps.

### Graphic card driver installation

The development of the project was based on the NVIDIA 535 driver. You may use a higher-version driver that is backward compatible with this version.

To install graphic driver, go to All Apps \> Software & Update \> Additional Drivers. Select Using Nvidia driver metapackage from nvidia-driver-535 (proprietary) and click Apply Changes.

![](media/image1.png)

Figure 1 NVIDIA deriver installation.

After installation, reboot the system to take effect. Check that your driver is installed correctly by opening a Terminal (Ctrl + Alt + T), and type

```
nvidia-smi
```

If you see the following results, the driver has been properly installed.

![](media/image2.png)

Figure 2 Sample output of nvidia-smi indicating a successful installation.

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

If you see similar output as below, docker has been successfully
installed.

![](media/image3.png)

Figure 3 Sample output of docker info indicating a successful
installation.

### NVIDIA docker installation

NVIDIA Docker, also known as NVIDIA Container Toolkit, is a set of tools and enhancements that enable GPU acceleration for Docker containers. This toolkit is necessary for the project. To install, follow the steps [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

