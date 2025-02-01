FROM ubuntu:22.04

COPY setup.sh /setup.sh

RUN chmod +x /setup.sh

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York
RUN apt update && apt install tzdata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN /setup.sh
COPY quaternions.py /usr/lib/python3/dist-packages/transforms3d/quaternions.py
