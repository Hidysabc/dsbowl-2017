from __future__ import print_function
from __future__ import division
import numpy as np
import os
import logging
import pandas as pd
from keras.models import Sequential, load_model
import boto3
from resnet3d import Resnet3DBuilder
import sys
import glob

s3bucket = "dsbowl2017-stage2-images"
s3bucket_model = "dsbowl2017-stage1-images"
input_sample_images = "sample_images"
input_preprocessing_images = "preprocessing_images"
input_csv = "csv"
input_dir = '/tmp/data/'
batch_size = 4
NB_CLASSES = 2
nb_epoch = 100

model_name = "hw-resnet3d-18"

image_path = os.path.join(input_dir, input_preprocessing_images)
output_filepath = os.path.join(input_dir,"prediction.csv")
FORMAT = '%(asctime)-15s %(name)-8s %(levelname)s %(message)s'
LOG_MAP = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "warn": logging.WARNING
        }

logging.basicConfig(format=FORMAT)
logger = logging.getLogger(model_name)

logger.setLevel(LOG_MAP["debug"])


s3 = boto3.resource('s3')
bucket = s3.Bucket(s3bucket)
s3_client = boto3.client('s3')

def generate_predict_data_from_directory(images, batch_size, image_path=image_path):
    while 1:
        for i in np.arange(int(len(images)/batch_size+0.5)):
            imgs = images[i*batch_size:(i+1)*batch_size]
            #imgs_id = p_images_id[i*batch_size:(i+1)*batch_size]
            img_array = np.array([np.load(os.path.join(image_path, j)) for j in imgs])
            #print((len(img_array)))
            yield (np.reshape(img_array, (img_array.shape[0],img_array.shape[1],img_array.shape[2], img_array.shape[3],1)))

logger.info("Start loading model...")
model_path = os.path.join(input_dir, sys.argv[1])
s3_client.download_file(s3bucket_model, sys.argv[1], model_path)
model = load_model(model_path)
predict_images = [os.path.basename(f) for f in glob.glob(os.path.join(image_path,'*.npy'))]
predict_gen = generate_predict_data_from_directory(predict_images, batch_size, image_path)


logger.info("Start predicting...")
prediction = model.predict_generator(predict_gen,
                                  steps = int(len(predict_images)/batch_size+0.5),
                                  max_q_size = 10,
                                  workers = 1,
                                  pickle_safe = False,
                                  verbose = 0)

logger.info("Finish Prediction! :)")
output = pd.DataFrame.from_dict({'id': [j.replace('.npy','') for j in predict_images], "cancer": prediction[:,1]})
output = output[["id","cancer"]]
output.to_csv(output_filepath,index = False)
s3_client.upload_file(output_filepath, s3bucket, os.path.basename(output_filepath))
logger.debug("Successfully upload prediciotn csv file {} to S3! ".format(os.path.basename(output_filepath)))


