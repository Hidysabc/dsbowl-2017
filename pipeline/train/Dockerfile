#####
#
# Docker file to build image that can run Data Science Bowl 2017 tutorial:
# U-Net Segmentation Approach to Cancer Diagnosis
#

FROM tensorflow/tensorflow:latest-gpu

MAINTAINER Wei-Yi Cheng

RUN pip install keras==1.2.2 pandas boto3 h5py && \
    rm -rf /root/.cache/pip/*

