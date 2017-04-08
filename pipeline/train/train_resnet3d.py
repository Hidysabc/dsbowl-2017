from __future__ import print_function
from __future__ import division
import numpy as np
import os
import logging
import pandas as pd
from scipy import stats
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten
from keras.layers.convolutional import Convolution3D
from keras.layers.pooling import MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, Adam
import boto3
from resnet3d import Resnet3DBuilder
import sys
from callbacks import ModelCheckpointS3

model_name = "hw-resnet3d-18"

#s3bucket = "dsbowl2017-sample-images"
s3bucket = "dsbowl2017-stage1-images"
input_sample_images = "sample_images"
input_preprocessing_images = "preprocessing_images"
input_csv = "csv"
input_dir = '/tmp/'
batch_size = 4
NB_CLASSES = 2
train_ratio = 0.90
validate_ratio = 0.10
nb_epoch = 100

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
all_keys = [obj.key for obj in bucket.objects.all()]
all_keys = [i for i in all_keys if input_preprocessing_images in i]
patient_id_keys = [filename.split('/') for filename in all_keys]
patient_id_keys = [i[1] for i in patient_id_keys if i[1]!=""]
patient_id = np.unique(patient_id_keys)
patient_id.sort()
patient_ids = [patient.replace('.npy','') for patient in patient_id]

csv_keys = [obj.key for obj in bucket.objects.all()]
csv_keys = [i for i in csv_keys if input_csv in i]
csv_info = [i for i in csv_keys if "sample_img_info" in i]

s3_client = boto3.client('s3')

'''
Don't do download here. Try EBS volume...20170408 Hidy chiu

if not os.path.exists(os.path.join(input_dir, csv_info[0])):
    s3_client.download_file(s3bucket, csv_info[0], os.path.join(input_dir,csv_info[0]))

if not os.path.exists(os.path.join(input_dir, input_preprocessing_images)):
    os.mkdir(os.path.join(input_dir,input_preprocessing_images))
    [s3_client.download_file(s3bucket, os.path.join(input_preprocessing_images,patient), os.path.join(input_dir,input_preprocessing_images,patient)) for patient in patient_id]

'''

'''
labels_info = pd.read_csv('/preprocessing_images_all/sample_images_.csv',
                          index_col=0)
'''
'''
labels_info = pd.read_csv(os.path.join(input_dir, csv_info[0]))
CANCER_MAP = labels_info.set_index('id')['cancer'].to_dict()
labels_info.set_index("id", drop=True, inplace=True)
'''

image_path = os.path.join(input_dir, input_preprocessing_images)
csv_path = os.path.join(input_dir,csv_info[0])
#images = [f for f in os.listdir(path) if f.endswith('.npy')]
#images = [f for f in images if f.replace(".npy","") in labels_info.index]

def split_data(image_path, csv_path, train_ratio=0.85, validate_ratio=0.05):
    images = [f for f in os.listdir(image_path) if f.endswith('.npy')]
    labels_info = pd.read_csv(csv_path)
    CANCER_MAP = labels_info.set_index('id')['cancer'].to_dict()
    labels_info.set_index("id", drop=True, inplace = True)
    images = [f for f in images if f.replace(".npy","") in labels_info.index]
    cancer_images = [f for f in images if CANCER_MAP[f.replace(".npy","")]==1]
    noncancer_images = [f for f in images if f not in cancer_images]
    train_cancer_images = cancer_images[:int(len(cancer_images)*train_ratio+0.5)]
    validate_cancer_images = cancer_images[len(train_cancer_images):len(train_cancer_images)
                                           +int(len(cancer_images)*validate_ratio+0.5)]
    test_cancer_images = cancer_images[len(train_cancer_images)+len(validate_cancer_images)
                                       :len(train_cancer_images)+len(validate_cancer_images)
                                       +int(len(cancer_images)*(1-train_ratio-validate_ratio)+0.5)]
    train_noncancer_images = noncancer_images[:int(len(noncancer_images)*train_ratio+0.5)]
    validate_noncancer_images = noncancer_images[len(train_noncancer_images):len(train_noncancer_images)
                                           +int(len(noncancer_images)*validate_ratio+0.5)]
    test_noncancer_images = noncancer_images[len(train_noncancer_images)+len(validate_noncancer_images)
                                       :len(train_noncancer_images)+len(validate_noncancer_images)
                                       +int(len(noncancer_images)*(1-train_ratio-validate_ratio)+0.5)]
    train_images = train_cancer_images+train_noncancer_images
    validate_images = validate_cancer_images+validate_noncancer_images
    test_images = test_cancer_images+test_noncancer_images

    return (train_images, validate_images, test_images)

def generate_data_from_directory(images, csv_path, batch_size, nb_classes=NB_CLASSES):
    #images = [f for f in os.listdir(image_path) if f.endswith('.npy')]
    labels_info = pd.read_csv(os.path.join(input_dir, csv_path))
    CANCER_MAP = labels_info.set_index('id')['cancer'].to_dict()
    labels_info.set_index("id", drop=True, inplace=True)
    #images = [f for f in images if f.replace(".npy","") in labels_info.index]
    ##images_id = [f.replace(".npy","") for f in images]
    while 1:
        #permute index
        p_images = np.random.permutation(images)
        p_images_id = [f.replace(".npy","") for f in p_images]
        for i in np.arange(int(len(images)/batch_size+0.5)):
            imgs = p_images[i*batch_size:(i+1)*batch_size]
            imgs_id = p_images_id[i*batch_size:(i+1)*batch_size]
            img_array = np.array([np.load(os.path.join(image_path, j)) for j in imgs])
            #print((len(img_array)))
            label = np_utils.to_categorical([CANCER_MAP[x] for x in imgs_id], nb_classes)
            #[print(l) for l in label]
            yield (np.reshape(img_array, (img_array.shape[0],img_array.shape[1],img_array.shape[2], img_array.shape[3],1)),label)


'''
tmp = [(img.replace('.npy', ''), np.load(os.path.join(path, img))) for img in images]
tmp = [img for img in tmp if img[0] in CANCER_MAP]
X = np.array([x[1] for x in tmp])
X = X.reshape((X.shape[0], X.shape[1], X.shape[2], X.shape[3], 1))
y = np_utils.to_categorical([CANCER_MAP[c[0]] for c in tmp])
'''

if len(sys.argv)>1:
    logger.info("Start loading model...")
    model_path = os.path.join(input_dir, sys.argv[1])
    s3_client.download_file(s3bucket, sys.argv[1], model_path)
    model = load_model(model_path)
else:
    logger.info("Start building model...")
    model = Resnet3DBuilder.build_resnet3D_18((1,128,256,256),2)


model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])


train_images, validate_images, test_images = split_data(image_path, csv_path, train_ratio=train_ratio, validate_ratio=validate_ratio)
train_gen = generate_data_from_directory(train_images, csv_path, batch_size)
validate_gen = generate_data_from_directory(validate_images, csv_path, batch_size)

checkpointer = ModelCheckpointS3(monitor='val_loss',filepath="/tmp/models.hdf5",
                                 bucket = s3bucket,
                                 verbose=0, save_best_only=True)

history = model.fit_generator(train_gen,
                              steps_per_epoch = int(len(train_images)/batch_size+0.5),
                              epochs = nb_epoch,
                              verbose = 1,
                              validation_data= validate_gen,
                              validation_steps = int(len(validate_images)/batch_size+0.5),
                              callbacks= [checkpointer])

model.save("{}.h5".format(model_name))
logger.info("Finish! :)")
#history = model.fit(X, y, nb_epoch=3, batch_size=2)


