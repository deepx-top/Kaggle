# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from load_res_mnist import load_res_mnist
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

batch_size = 16
img_rows, img_cols, img_channel = 224, 224, 3
num_classes = 10
epochs = 2
# train = load_res_mnist('./train.csv')
# x_ = train.images
# y_ = train.labels

dgen = ImageDataGenerator(rescale=1. / 255)
train_gen = dgen.flow_from_directory('./train/', target_size=(
    224, 224), batch_size=batch_size, class_mode='categorical', shuffle=True)

inp = Input(batch_shape=(None, img_rows, img_cols,
                         img_channel), name='input_image')
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inp)
boo = base_model.output
boo = Flatten()(boo)
dense_1 = Dense(512, activation='relu')(boo)
dense_2 = Dense(512, activation='relu')(dense_1)
out_cls = Dense(num_classes, activation='softmax')(dense_2)

model = Model(inputs=inp, outputs=out_cls)
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy', metrics=['accuracy'])

# datagen = ImageDataGenerator()
# datagen.flow(x_train, y_train, batch_size=5)

model.fit_generator(train_gen, steps_per_epoch=200, epochs=1, verbose=1)
# # predictions = model.predict(test, batch_size=5, verbose=1)


test_gen = dgen.flow_from_directory('./test/', target_size=(
    224, 224), batch_size=batch_size, class_mode=None)

yPred = model.predict_generator(test_gen, verbose=1)

np.savetxt('mnist_output_vgg16.csv', np.c_[range(1, len(
    yPred) + 1), yPred], delimiter=',', header='ImageId,Label', comments='', fmt='%d')
