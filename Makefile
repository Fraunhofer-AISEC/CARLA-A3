
build-ue-carla:
	bash DockerBuildScript/BuildEntry.sh

# Use the below when running carla after initial run
run-ue-carla: NAME=carla-ue
run-ue-carla: IMAGE=carla
run-ue-carla: ENTRYCMD=make launch-only
run-ue-carla: EXTRAVOLUME=-v "${PWD}/CarlaVolume/carla:/home/carla/carla"
run-ue-carla: run



# Use the below when running a new carla build for the first time
run-init-carla: NAME=carla-ue
run-init-carla: IMAGE=carla
run-init-carla: ENTRYCMD=make launch
run-init-carla: EXTRAVOLUME=-v "${PWD}/CarlaVolume/carla:/home/carla/carla"
run-init-carla: run

.PHONY: run

run:
	docker run ${DFLAG} \
	-it \
	--rm \
	--name ${NAME} \
	--privileged \
	--runtime nvidia \
	--net=host \
	-e DISPLAY=${DISPLAY} \
	-v "${PWD}/workingdir:/home/carla/workingdir" \
	${EXTRAVOLUME} \
	${IMAGE} ${ENTRYCMD}

