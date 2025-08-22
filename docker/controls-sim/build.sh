# Unfortunately one of the more tragic Docker philosophies is its very very robust "context"
# The only solution I could find was to copy these required folders into the context
# If anyone can debug the Dockerfile to support building docker images from outside the context/
# from the home folder, that would be great :)

# But for now, we use the janky solution:

rm -rf "$DRIVERLESS/tmp_driverless_ws"
cp -r "$DRIVERLESS/driverless_ws" "$DRIVERLESS/docker/controls-sim/tmp_driverless_ws"

sudo docker build --platform linux -t controls-sim .
