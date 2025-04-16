# Build with 
# docker build --network=host -t carla-step0 -f step0.Dockerfile .

FROM nvidia/vulkan:1.1.121-cuda-10.1--ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=teletype

# The nvidia repo there causes GPG key error and the build process to fail, so remove it first
RUN rm /etc/apt/sources.list.d/* 
RUN apt-get update
RUN apt-get install -y wget nano

RUN useradd -m carla

WORKDIR /home/carla/

USER carla

# Copy dependencies into the image
COPY /dependencies/carla.tar.xz /home/carla/
RUN tar -xf carla.tar.xz
RUN rm carla.tar.xz
COPY --chown=carla:carla /dependencies/UnrealEngine /home/carla/UnrealEngine_4.26

