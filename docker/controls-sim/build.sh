# Unfortunately one of the more tragic Docker philosophies is its very very robust "context"
# The only solution I could find was to copy these required folders into the context
# If anyone can debug the Dockerfile to support building docker images from outside the context/
# from the home folder, that would be great :)

# But for now, we use the janky solution:
docker_root_path="$DRIVERLESS/docker/controls-sim"

rm -rf "$docker_root_path/tmp_driverless_ws"
cp -r "$DRIVERLESS/driverless_ws" "$docker_root_path/tmp_driverless_ws"

rm -rf "$docker_root_path/canUsbKvaserTesting"
cp -r "$LINUXCAN/" "$docker_root_path/canUsbKvaserTesting"

sudo docker build --platform linux -t controls-sim .
