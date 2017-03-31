#####
#
# Docker file to build image that can run Data Science Bowl 2017 tutorial:
# U-Net Segmentation Approach to Cancer Diagnosis
#

FROM continuumio/anaconda

MAINTAINER Hidy Chiu

RUN pip install boto3 pydicom && \
    rm -rf /root/.cache/pip/*

RUN cd /root && git clone https://github.com/Hidysabc/dsbowl-2017.git 

WORKDIR /root/dsbowl-2017/pipeline/preprocess
