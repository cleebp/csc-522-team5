import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings

import tensorflow as tf
from scipy.misc import toimage, imresize

from keras.models import Sequential, Model
from keras.layers import Dense, Convolution2D, Flatten, BatchNormalization, Convolution3D
from keras.layers import Activation, MaxPooling2D, UpSampling2D, Lambda, MaxPooling3D, Dropout
from keras.optimizers import Adam

from sklearn.metrics import log_loss

#%matplotlib inline
warnings.filterwarnings("ignore")


IMG_SIZE_PX = 256
SLICE_COUNT = 64

n_classes = 2

keep_rate = 0.8


def save_array(path, arr):
    np.save(path, arr)
    

def load_array(path):
    return np.load(path)


train_ids = [id.replace(".npy", "") for id in os.listdir('prepd_stage1/train/')]
valid_ids = [id.replace(".npy", "") for id in os.listdir('prepd_stage1/valid/')]
test_ids = [id.replace(".npy", "") for id in os.listdir('prepd_stage1/test/')]

train_ids.sort()
valid_ids.sort()
test_ids.sort()

print(train_ids[:3])
print(valid_ids[:3])
print(test_ids[:3])



df = pd.read_csv('sample_truth.csv', error_bad_lines=False)
print(df)
df.head()


train_labels = df["cancer"][df["id"].isin(train_ids)].values
valid_labels = df["cancer"][df["id"].isin(valid_ids)].values

print(train_labels[:3])
print(valid_labels[:3])



n_slices = 32

def get_train_batches():

    for ix, patient in enumerate(train_ids):
        sample = load_array("prepd_stage1/train/{}.npy".format(patient))
        sample = np.array([imresize(toimage(im), size=(IMG_SIZE_PX, IMG_SIZE_PX)) for im in sample])
        
        ht = sample.shape[0]
        bottom = int(np.floor((ht - SLICE_COUNT)/2))
        top = int(np.floor((ht + SLICE_COUNT)/2))
        
        # plt.imshow(sample[100], cmap="gray")
        
        try:
            sample = np.array([sample[bottom:top, :, :].reshape(1, SLICE_COUNT, IMG_SIZE_PX, IMG_SIZE_PX)])
            label = np.array([train_labels[ix]])    
            yield sample, label
            
        except Exception as e:
            print(patient, e, sample.shape)
            continue

            
def get_valid_batches():
    for ix, patient in enumerate(valid_ids):
        sample = load_array("prepd_stage1/valid/{}.npy".format(patient))
        
        sample = np.array([imresize(toimage(im), size=(IMG_SIZE_PX, IMG_SIZE_PX)) for im in sample])
        
        ht = sample.shape[0]
        bottom = int(np.floor((ht - SLICE_COUNT)/2))
        top = int(np.floor((ht + SLICE_COUNT)/2))
        
        try:
            sample = np.array([sample[bottom:top, :, :].reshape(1, SLICE_COUNT, IMG_SIZE_PX, IMG_SIZE_PX)])
            label = np.array([valid_labels[ix]])
            yield sample, label
            
        except Exception as e:
            print(patient, e, sample.shape)
            continue
            
            
def get_test_batches():
    for ix, patient in enumerate(test_ids):
        sample = load_array("prepd_stage1/test/{}.npy".format(patient))
        
        sample = np.array([imresize(toimage(im), size=(IMG_SIZE_PX, IMG_SIZE_PX)) for im in sample])
        
        ht = sample.shape[0]
        bottom = int(np.floor((ht - SLICE_COUNT)/2))
        top = int(np.floor((ht + SLICE_COUNT)/2))
        
        try:
            sample = np.array([sample[bottom:top, :, :].reshape(1, SLICE_COUNT, IMG_SIZE_PX, IMG_SIZE_PX)])
            yield sample
            
        except Exception as e:
            print(patient, e, sample.shape)
            continue



model = Sequential()

model.add(Convolution3D(32, 7, 7, 7, activation='tanh', border_mode='same', input_shape=(1, SLICE_COUNT, IMG_SIZE_PX, IMG_SIZE_PX)))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='same'))
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.5))

model.add(Convolution3D(64, 7, 7, 7, activation='tanh', border_mode='same'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='same'))
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation='tanh'))
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

optim = Adam(lr=.01)
model.compile(optimizer=optim, loss='binary_crossentropy')



epochs = 15

for e in range(epochs):
    print("\nEpoch {0} of {1}...".format(e+1, epochs))
    trn_gen = get_train_batches()
    val_gen = get_valid_batches()

    for X, y in trn_gen:
        model.train_on_batch(X, y)
        
    y_pred = []
    y_true = []
    for X, y in val_gen:
        prediction = model.predict(X, verbose=0)
        y_true.append(y[0])
        y_pred.append(prediction[0])

    print("Val Loss:", log_loss(y_true, y_pred))



model.save("3d_v1.h5")


test_gen = get_test_batches()

test_x = np.array([x for x in test_gen])
test_x = np.vstack(test_x)

test_x[:5]



