"""
CANNOT BE USED IN CPU, CAUSED CORE DUMP

Only supported in cudnn
"""

import logging

logger = logging.getLogger('mxnet-conv3d')
logger.setLevel(logging.DEBUG)

import mxnet as mx
import numpy as np
import os
import pandas as pd
import scipy
from scipy import stats
import skimage.transform as imgtransform


PRJ = '/workspace/dsbowl2017'
DATA_PATH = os.path.join(PRJ, 'data')
images_folder = os.path.join(DATA_PATH, 'preprocessing_images_spacing211')
patients = [f.replace('.npy', '') for f in os.listdir(images_folder)]


df_label = pd.read_csv(os.path.join(DATA_PATH, 'stage1_labels.csv'))

X = np.stack([imgtransform.resize(np.load(os.path.join(images_folder, f + '.npy')),
                                  (32, 64, 64), mode='edge') for f in patients
              if f in df_label.id.values])

X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2], X.shape[3])

y = df_label[df_label.id.isin(patients)].cancer.values

train_iter = mx.io.NDArrayIter(data=X, label=y, batch_size=5, shuffle=True)

net = mx.sym.Variable('data')
label = mx.sym.Variable('softmax_label')

net = mx.sym.Convolution(data=net, num_filter=32, kernel=(1, 3, 3), stride=(1, 2, 2))
net = mx.sym.BatchNorm(data=net)
net = mx.sym.Activation(data=net, act_type='relu')
net = mx.sym.Convolution(data=net, num_filter=32, kernel=(3, 3, 3), stride=(2, 2, 2))
net = mx.sym.BatchNorm(data=net)
net = mx.sym.Activation(data=net, act_type='relu')
net = mx.sym.Convolution(data=net, num_filter=16, kernel=(1, 3, 3), stride=(1, 2, 2))
net = mx.sym.BatchNorm(data=net)
net = mx.sym.Activation(data=net, act_type='relu')
net = mx.sym.Convolution(data=net, num_filter=16, kernel=(3, 3, 3), stride=(2, 2, 2))
net = mx.sym.BatchNorm(data=net)
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

model.fit(train_iter,
          optimizer='adam',
          optimizer_params={'learning_rate':0.01},
          eval_metric='acc',
          num_epoch=3)



