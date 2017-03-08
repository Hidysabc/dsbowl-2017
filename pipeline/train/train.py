import numpy as np
import os
import pandas as pd
from scipy import stats
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten
from keras.layers.convolutional import Convolution3D
from keras.layers.pooling import MaxPooling3D
from keras.optimizers import RMSprop, Adam

labels_info = pd.read_csv('/preprocessing_images_all/sample_images_.csv',
                          index_col=0)
CANCER_MAP = labels_info.set_index('id')['cancer'].to_dict()
path = '/preprocessing_images_all/'
images = [f for f in os.listdir(path) if f.endswith('.npy')]
tmp = [(img.replace('.npy', ''), np.load(os.path.join(path, img))) for img in images]
tmp = [img for img in tmp if img[0] in CANCER_MAP]
X = np.array([x[1] for x in tmp])
X = X.reshape((X.shape[0], X.shape[1], X.shape[2], X.shape[3], 1))
y = np_utils.to_categorical([CANCER_MAP[c[0]] for c in tmp])

model = Sequential()
model.add(Convolution3D(32, 3, 3, 3, border_mode='same',activation='relu', input_shape=(128,256,256, 1)))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Convolution3D(32, 3, 3, 3, border_mode='same',activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Convolution3D(16, 3, 3, 3, border_mode='same',activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Convolution3D(16, 3, 3, 3, border_mode='same',activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Convolution3D(8, 3, 3, 3, border_mode='same',activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Convolution3D(8, 3, 3, 3, border_mode='same',activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit(X, y, nb_epoch=3, batch_size=2)


