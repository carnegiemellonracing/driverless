xhost + 
docker run -dt --name cmrdv_docker --restart unless-stopped -e DISPLAY=docker.for.mac.host.internal:0 -v `pwd`:/root/workspace gracetangg/cmrdv_image:v5-release
