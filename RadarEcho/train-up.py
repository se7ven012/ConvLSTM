# %%
import numpy as np
import os
from keras.models import Sequential
from keras.layers.convolutional import Conv3D, Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras.optimizers import Adadelta, Adam
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
from math import sqrt

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

# use one frame to predict next frame
def shift_data(data, n_frames):
    X = data[:, 0:n_frames, :, :, :]
    y = data[:, 1:(n_frames+1), :, :, :]
    return X, y

# Define network
def network(gpus):
    seq = Sequential()
    seq.add(ConvLSTM2D(
                    filters=64, 
                    kernel_size=(1,1),
                    input_shape=(None, image_height, image_width, 3), #Will need to change channels to 3 for real images
                    padding='same', 
                    return_sequences=True,
                    activation='relu'))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(
                    filters=32, 
                    kernel_size=(1,1),
                    padding='same', 
                    return_sequences=True,
                    activation='relu'))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(
                    filters=32, 
                    kernel_size=(1,1),
                    padding='same', 
                    return_sequences=True,
                    activation='relu'))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(
                    filters=64, 
                    kernel_size=(1,1),
                    padding='same', 
                    return_sequences=True,
                    activation='relu'))
    seq.add(BatchNormalization())

    seq.add(Conv3D(
                    filters=3, 
                    kernel_size=(1, 1, 1),
                    activation='sigmoid',
                    padding='same', 
                    data_format='channels_last'))

    parallel_mode = multi_gpu_model(seq, gpus)
    parallel_mode.compile(loss=root_mean_squared_error, optimizer=Adam(1e-3))

    print(parallel_mode.summary())
    return parallel_mode

# load dataset
# 数据集在预处理时已经归一化，这里就不用归一了
datapath=['./data/A_rgb1','./data/A_rgb2','./data/A_rgb3','./data/A_rgb4']
data = np.concatenate((np.load(datapath[0]+'/dataset.npy'),np.load(datapath[1]+'/dataset.npy')),axis=0)
data = np.concatenate((data,np.load(datapath[2]+'/dataset.npy')),axis=0)
data = np.concatenate((data,np.load(datapath[3]+'/dataset.npy')),axis=0)

# Define image #frame,hight,width
input_frame,image_height,image_width = 11,data.shape[2],data.shape[3]

# X/y.shape=(xx, input_frame, 128, 128, 3) 
# 样本数，样本帧数，图宽，图高，颜色通道数
X, y = shift_data(data, input_frame)
# Random shuffle
index = [i for i in range(len(data))]
np.random.shuffle(index) 
X = X[index]
y = y[index]

# Start train
gpus = 2
seq = network(gpus)

log_dir = 'logs/'
logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=3, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

seq.fit(X, y, batch_size=4, epochs=40, validation_split=0.05, callbacks=[logging, checkpoint, reduce_lr, early_stopping])
seq.save('Full_128_A_'+'40'+'.h5')

# %%
