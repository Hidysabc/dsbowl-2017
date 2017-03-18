#####
#
# Docker file to build image that can run Data Science Bowl 2017 tutorial:
# U-Net Segmentation Approach to Cancer Diagnosis
#

FROM dmlc/mxnet:cuda

MAINTAINER Wei-Yi Cheng

RUN pip install SimpleITK jupyter && \
    # install tensorflow GPU
    #pip install tensorflow-gpu &&\
    # install tensorflow CPU
    #-- clean up --
    rm -rf /root/.cache/pip/*

