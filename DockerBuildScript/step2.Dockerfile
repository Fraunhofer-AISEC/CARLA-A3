## step2 - Build UE4
# Build with
# docker build -t carla-step2 -f step2.Dockerfile .

FROM carla-step1

USER carla
ENV UE4_ROOT=/home/carla/UnrealEngine_4.26

RUN cd $UE4_ROOT && \
  ./Setup.sh && \
  ./GenerateProjectFiles.sh && \
  make
WORKDIR /home/carla/
