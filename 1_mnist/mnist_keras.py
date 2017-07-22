# -*- coding:utf-8 -*-

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from load_mnist import load_mnist
from keras.preprocessing.image import ImageDataGenerator
from figures_save import figures

batch_size = 50
img_rows, img_cols = 28, 28
num_classes = 10
epochs = 5
train = load_mnist('./train.csv')
trainx = train.images.reshape(-1, img_rows, img_cols, 1)
trainy = train.labels

input_shape = (img_rows, img_cols, 1)
# model
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.0008),
              metrics=['accuracy'])

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=10,
    zoom_range=0.1,  # Randomly zoom image
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

# Fit the model
history = model.fit_generator(datagen.flow(trainx, trainy, batch_size=batch_size),
                              epochs=epochs, verbose=2, steps_per_epoch=int(trainx.shape[0] / batch_size))

figures(history)
# model.fit(trainx, trainy,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_split=0.30)


test = load_mnist('./test.csv')
testx = test.images.reshape(-1, img_rows, img_cols, 1)
result = model.predict_classes(testx, verbose=0)

pd.DataFrame({"ImageId": list(range(1, len(result) + 1)),
              "Label": result}).to_csv("submission_keras2.csv",
                                       index=False, header=True)
