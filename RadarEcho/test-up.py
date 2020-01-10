# %%
import numpy as np
import os
import skimage
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from math import sqrt

datapath='./test/dataset.npy'

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

# use one frame to predict next frame
def shift_data(data, n_frames):
    X = data[:, 0:n_frames, :, :, :]
    y = data[:, 1:(n_frames+1), :, :, :]
    return X, y

# data.shape=(440, 12, 128, 128, 3)
data = np.load(datapath)

# Define image #frame,hight,width
input_frame,image_height,image_width = 11,data.shape[2],data.shape[3]

# X/y.shape=(434, 12, 128, 128, 3)
X, y = shift_data(data, input_frame)
# Random shuffle
index = [i for i in range(len(data))]
np.random.shuffle(index) 
X = X[index]
y = y[index]

test_set = np.expand_dims(X[1, :, :, :, :], 0)
#test_set.shape=(1, 12, 128, 128, 3)

# load model
seq = load_model('Full_128_A_'+'20'+'.h5',custom_objects={'root_mean_squared_error':root_mean_squared_error})
prediction = seq.predict(test_set)

# visualize result
for i in range(0, 11):
    # create plot
    fig = plt.figure(figsize=(10, 5))

    # test
    ax = fig.add_subplot(122)
    ax.text(1, -3, ('ground truth at time :' + str(i)), fontsize=20, color='b')
    toplot_true = test_set[0, i, ::, ::, 0]
    plt.imshow(toplot_true)

    # predictions
    ax = fig.add_subplot(121)
    ax.text(1, -3, ('predicted frame at time :' + str(i)), fontsize=20, color='b')
    toplot_pred = prediction[0, i, ::, ::, 0]
    plt.imshow(toplot_pred)

    print(sqrt(mean_squared_error(toplot_true, toplot_pred)))
    plt.savefig('image'+'_'+str(data.shape[2])+'_'+str(data.shape[3])+'_'+str(i)+'.png')

# %%
