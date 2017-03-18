"""
CANNOT BE USED IN CPU, CAUSED CORE DUMP

Only supported in cudnn
"""

from __future__ import print_function
from __future__ import division
import logging
import mxnet as mx
from mxnet.callback import do_checkpoint, Speedometer
import numpy as np
import os
import pandas as pd
import boto3

model_name = 'mxnet-conv3d'
s3bucket = "dsbowl2017-sample-images"
input_sample_images = "sample_images"
input_preprocessing_images = "preprocessing_images"
input_csv = "csv"
input_dir = '/tmp/'
model_dir = '/tmp/model'
batch_size = 5
NB_CLASSES = 2

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


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

if not os.path.exists(os.path.join(input_dir, csv_info[0])):
    s3_client.download_file(s3bucket, csv_info[0], os.path.join(input_dir,csv_info[0]))

if not os.path.exists(os.path.join(input_dir, input_preprocessing_images)):
    os.mkdir(os.path.join(input_dir, input_preprocessing_images))
    _ = [s3_client.download_file(s3bucket, os.path.join(input_preprocessing_images,patient),
         os.path.join(input_dir,input_preprocessing_images,patient)) for patient in patient_id]

image_path = os.path.join(input_dir, input_preprocessing_images)
csv_path = os.path.join(input_dir,csv_info[0])

df_label = pd.read_csv(os.path.join(input_dir, csv_path))
cancer_map = df_label.set_index('id')['cancer'].to_dict()
df_label.set_index('id', drop=True, inplace=True)
images = [f for f in os.listdir(image_path) if f.endswith('.npy')]
images_id = [f.replace('.npy', '') for f in images]
images = [f for f in images if f.replace('.npy', '') in df_label.index]

X = np.stack([np.load(os.path.join(image_path, j)) for j in images])
X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2], X.shape[3])

y = df_label[df_label.index.to_series().isin(images_id)].cancer.values

train_iter = mx.io.NDArrayIter(data=X, label=y, batch_size=5, shuffle=True)

net = mx.sym.Variable('data')
label = mx.sym.Variable('softmax_label')

net = mx.sym.Convolution(data=net, num_filter=32, kernel=(1, 3, 3), stride=(1, 2, 2))
net = mx.sym.Activation(data=net, act_type='relu')
net = mx.sym.Convolution(data=net, num_filter=32, kernel=(3, 3, 3), stride=(2, 2, 2))
net = mx.sym.Activation(data=net, act_type='relu')
net = mx.sym.Convolution(data=net, num_filter=16, kernel=(1, 3, 3), stride=(1, 2, 2))
net = mx.sym.Activation(data=net, act_type='relu')
net = mx.sym.Convolution(data=net, num_filter=16, kernel=(3, 3, 3), stride=(2, 2, 2))
net = mx.sym.Activation(data=net, act_type='relu')
net = mx.sym.FullyConnected(data=net, num_hidden=64)
net = mx.sym.SoftmaxOutput(data=net, name='softmax', label=label)

#mx.viz.plot_network(net, shape={'data': X.shape})

model = mx.mod.Module(
    symbol = net,       # network structure
    context = mx.gpu(),
    data_names = ['data'],
    label_names = ['softmax_label']
)

batch_end_callback = mx.callback.Speedometer(batch_size, frequent=50)

epoch_end_callback = mx.callback.do_checkpoint(
    '{}/{}'.format(model_dir, model_name),
    period=1)

model.fit(train_iter,
          optimizer='adam',
          optimizer_params = {'learning_rate':0.01},
          epoch_end_callback = epoch_end_callback,
          batch_end_callback = batch_end_callback,
          eval_metric='acc',
          num_epoch=3)



