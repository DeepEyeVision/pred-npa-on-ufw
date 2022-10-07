#!/bin/sh

docker run \
-it \
--rm \
--mount \
type=bind,\
source=/home/ai1/hisadome/pred-npa-on-ufw, \
target=/work \
-v /etc/passwd:/etc/passwd:ro \
-v /etc/group:/etc/group:ro \
-v /dev/bus/usb:/dev/bus/usb:ro \
--device /dev/dri:/dev/dri \
--device-cgroup-rule='c 189:* rmw' \
--cpuset-cpus 1 \
--name vino \
vino-q fish
