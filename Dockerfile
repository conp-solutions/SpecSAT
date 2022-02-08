FROM ubuntu:20.04

RUN apt-get update && \
    apt-get -y upgrade && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install \
        gcc \
        git \
        g++ \
        make \
        python3 \
        python3-pip

ADD ./requirements.txt /opt/requirements.txt

RUN python3 -m pip install --upgrade -r /opt/requirements.txt
