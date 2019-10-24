#%%
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import matplotlib.pyplot as plt

# Define network
def network():
    seq = Sequential()
    seq.add(ConvLSTM2D(
                    filters=64, 
                    kernel_size=(1,1),
                    input_shape=(None, image_height, image_width, 1), #Will need to change channels to 3 for real images
                    padding='same', 
                    return_sequences=True,
                    activation='relu'))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(
                    filters=64, 
                    kernel_size=(2,2),
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

    seq.add(ConvLSTM2D(
                    filters=64, 
                    kernel_size=(2,2),
                    padding='same', 
                    return_sequences=True,
                    activation='relu'))
    seq.add(BatchNormalization())

    seq.add(Conv3D(
                    filters=1, 
                    kernel_size=(1,1,1),
                    activation='sigmoid',
                    padding='same', 
                    data_format='channels_last'))
    seq.compile(loss='binary_crossentropy', optimizer='adam')
    return seq


#data.shape = (20, 10000, 64, 64)
data = np.load('mnist_test_seq.npy')

# Define image #frame,hight,width
frame,image_height,image_width = 15,data.shape[2],data.shape[3]

# 矩阵旋转维度
# swap frames and observations so [obs, frames, height, width, channels]
# data.shape = (10000, 20, 64, 64)
data = data.swapaxes(0, 1)

# only select first 100 observations to reduce memory- and compute requirements
sub = data[:100, :, :, :]

# add channel dimension (grayscale)
# data.shape = (10000, 20, 64, 64, 1)
sub = np.expand_dims(sub, 4)

# normalize to 0, 1
#sub = sub / 255
sub[sub < 128] = 0
sub[sub>= 128] = 1


# Add helper function for shifting input and output, 
# so previous frame (X_t-1) is used as input to predict next frame (y_t)
def shift_data(data, n_frames):
    X = data[:, 0:n_frames, :, :, :]
    y = data[:, 1:(n_frames+1), :, :, :]
    return X, y

# X.shape=(100, 15, 64, 64, 1)
X, y = shift_data(sub, frame)

seq = network()
seq.fit(X, y, batch_size=10, epochs=1, validation_split=0.05)

#X[5, :, :, :, :].shape=(15, 64, 64, 1)
# test_set.shape=(1, 15, 64, 64, 1)
test_set = np.expand_dims(X[5, :, :, :, :], 0)
prediction = seq.predict(test_set)

# visualize result
for i in range(0, 13):
    # create plot
    fig = plt.figure(figsize=(10, 5))

    # truth
    ax = fig.add_subplot(122)
    ax.text(1, -3, ('ground truth at time :' + str(i)), fontsize=20, color='b')
    toplot_true = test_set[0, i, ::, ::, 0]
    plt.imshow(toplot_true)

    # predictions
    ax = fig.add_subplot(121)
    ax.text(1, -3, ('predicted frame at time :' + str(i)), fontsize=20, color='b')
    toplot_pred = prediction[0, i+1, ::, ::, 0]
    plt.imshow(toplot_pred)

    plt.savefig('image'+str(i)+'.png')