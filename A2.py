from tqdm import tqdm
from PIL import Image
import pandas as pd
import os
import numpy as np


def load_data(folder):
    images = []
    # open up all directories in the input folder
    for file in os.listdir(folder):
        # extract file names without the extension
        file_id = file.replace('.png', '')
        # open the image
        image = Image.open(
            # make path with folder as root for each file
            os.path.join(folder, file)
            # convert using LA and resize so all images are the same dimension
        ).convert('LA').resize((256, 256))
        arr = np.array(image)
        images.append(
            # list with id and array
            (int(file_id), arr)
        )
    # sort images by id
    images.sort(key=lambda i: i[0])
    # doesn't return ids
    return np.array([v for _id, v in images])



x_train = load_data('train')
y_train = pd.read_csv('y_train.csv')['infection']

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Input, MaxPool2D, AveragePooling2D


# build fully connected NN
def build():
    # the dimensions of the image we expect, input for tensorflow
    img_in = Input(shape=(256, 256, 2))
    # reshapes to 1D
    flattened = Flatten()(img_in)
    # make dense MPL
    fc1 = Dense(80)(flattened)

    # gets rid of nodes below given threshold
    #fc1 = Dropout(0.1)(fc1)

    # make another layer of fully connected neurons
    fc2 = Dense(80)(fc1)

    #fc2 = Dropout(0.1)(fc2)
    # output layer, sigmoid activation
    output = Dense(1, activation = 'sigmoid')(fc2)
    # make model
    model = tf.keras.Model(inputs=img_in, outputs=output)
    return model

# builds convolutional NN
def buildConv():
    # image is 256x256x2
    img_in = Input(shape=(256, 256, 2))
    # perform convolution twice both using relu, add padding so that no pixels missed
    conv = Conv2D(filters=16, kernel_size=3, activation='relu', padding='same')(img_in)
    conv = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')(conv)
    # pooling to reduce amount of info, helps with speed
    conv = MaxPool2D(pool_size=(2, 2))(conv)

    # do a set of bigger layers, followed by pooling
    conv = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(conv)
    conv = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(conv)
    conv = MaxPool2D(pool_size=(2, 2))(conv)

    conv = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(conv)
    conv = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(conv)
    conv = MaxPool2D(pool_size=(2, 2))(conv)

    conv = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(conv)
    conv = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(conv)
    conv = MaxPool2D(pool_size=(2, 2))(conv)

    conv = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(conv)
    conv = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(conv)
    conv = MaxPool2D(pool_size=(2, 2))(conv)

    conv = Flatten()(conv)

    #conv = Dense(100, activation='relu')(conv)
    out = Dense(1, activation='sigmoid')(conv)

    # make model, return it
    model = tf.keras.Model(inputs=img_in, outputs=out)
    return model


# shuffle first - for extra question.
# rng_state = np.random.get_state()
# np.random.shuffle(x_train)
# np.random.set_state(rng_state)
# np.random.shuffle(y_train)

model = buildConv()
# model = build()


model.compile(
    # use adam optimizer (tried others, in doc)
    optimizer=tf.keras.optimizers.Adam(),
    # loss function, defines errors of model, how its evaluated
    loss='binary_crossentropy',
    # metrics to be evaluated in training and testing
    # binaryAccuracy = cross entropy for binary classes (0 and 1)
    # AUC for area under curve, involves false pos/neg and tru pos/neg
    metrics=['BinaryAccuracy', 'AUC']
    )

# prints out summary of the model
model.summary()

# define epoch and batch size
epochs = 20
batch_size = 64

history = model.fit(x = x_train,
                    y = y_train,
                    batch_size = batch_size,
                    validation_split=0.3, # validation is 30%
                    epochs=epochs
                    )

x_test = load_data('test')

y_test = model.predict(x_test)

y_train2 = model.predict(x_train)


y_train2_df = pd.DataFrame()
y_train2_df['id'] = np.arange(len(y_train2))
y_train2_df['infection'] = y_train2.astype(float)
y_train2_df.to_csv('train.csv', index=False)

y_test_df = pd.DataFrame()
y_test_df['id'] = np.arange(len(y_test))
y_test_df['infection'] = y_test.astype(float)
y_test_df.to_csv('test.csv', index=False)