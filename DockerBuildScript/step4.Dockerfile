FROM carla-step3

# Build with
# docker build -t carla-step4 -f step4.Dockerfile .

USER carla
WORKDIR /home/carla/carla

RUN make build.utils

RUN make package && \
    rm -r /home/carla/carla/Dist

WORKDIR /home/carla/carla
