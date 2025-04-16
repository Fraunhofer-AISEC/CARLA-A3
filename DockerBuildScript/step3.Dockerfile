FROM carla-step2

# Build with
# docker build -t carla-step3 -f step3.Dockerfile .

USER carla
WORKDIR /home/carla/carla

RUN ./Update.sh  && \
    make CarlaUE4Editor && \
    make PythonAPI

WORKDIR /home/carla/carla