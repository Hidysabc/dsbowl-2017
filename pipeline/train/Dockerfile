#####
#
# Docker file to build image that can run Data Science Bowl 2017 tutorial:
# U-Net Segmentation Approach to Cancer Diagnosis
#

FROM tensorflow/tensorflow:latest-gpu

MAINTAINER Wei-Yi Cheng, Hidy Chiu

RUN apt-get update && apt-get install -y git-all

RUN pip install keras==1.2.2 pandas boto3 h5py && \
    rm -rf /root/.cache/pip/*

RUN cd /root && git clone https://github.com/Hidysabc/dsbowl-2017.git

WORKDIR /root/dsbowl-2017/pipeline/train
