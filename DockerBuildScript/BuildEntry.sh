#! /bin/bash

# change ROOT if needed

SCRIPTDIR=$(dirname $(realpath $0))
echo SCRIPTDIR $SCRIPTDIR
ROOTDIR=$(dirname $SCRIPTDIR)
echo ROOTDIR $ROOTDIR

cd $SCRIPTDIR

docker build -t carla-step0 -f step0.Dockerfile .
docker build -t carla-step1 -f step1.Dockerfile .
docker build -t carla-step2 -f step2.Dockerfile .
docker build -t carla-step3 -f step3.Dockerfile .
docker build -t carla-step4 -f step4.Dockerfile .
docker build -t carla-step5 -f step5.Dockerfile .
docker build -t carla-step6 -f step6.Dockerfile .
docker build -t carla-step7 -f step7.Dockerfile .
docker build -t carla-step8 -f step8.Dockerfile .
docker tag carla-step8 carla
