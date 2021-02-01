FROM ubuntu:18.04 as builder
MAINTAINER Ramin Ranjbar "raxbits@gmail.com"

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get dist-upgrade -y && \
    apt-get install -y \
        udev \
        libx11-6 \
        libgomp1 python3-pip python3-dev libsm6 libxext6 libxrender-dev libusb-1.0-0 libcap2 \
        ca-certificates \
        cmake \
        build-essential \
        git \
        libx11-dev \
        udev \
        wget \
        libglib2.0-0

RUN apt-get install -y libgl1-mesa-glx

# RUN cd /tmp && \
#     wget https://en.ids-imaging.com/download-ueye-lin64.html?file=tl_files/downloads/uEye_SDK/driver/ids-software-suite-linux-4.93.0-64.tgz -O uEye-linux-4.93.0-64.tgz && \
#     tar xvzf uEye-linux-4.93.0-64.tgz && chmod +x *.run && \
#     yes | ./ueye_4.93.0.989_amd64.run --nox11
RUN mkdir -p /app
COPY requirements.txt /app
WORKDIR /app
run pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
COPY . /app
# ENTRYPOINT ["python3", "main.py"]
ENTRYPOINT ["bash","run.sh"]
