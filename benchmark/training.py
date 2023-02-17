import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import cv2
import os
from mobileNet_v1 import mobilenet_v1


IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 200
batch_size = 50

BASE_DIR = os.path.join(os.getcwd(), 'vw_coco2014_96')
validation_split = 0.1

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=.1,
    horizontal_flip=True,
    validation_split=validation_split,
    rescale=1. / 255)
train_generator = datagen.flow_from_directory(
    BASE_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='training',
    color_mode='rgb')
val_generator = datagen.flow_from_directory(
    BASE_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation',
    color_mode='rgb')
print(train_generator.class_indices)

#%%
model = mobilenet_v1()
epochs = 150
learning_rate = 0.0001

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=len(val_generator),
                    batch_size=BATCH_SIZE)

model.save('mobilenet_trained_model')
acc = history.history['accuracy']
loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
results = np.concatenate((acc, loss, val_acc, val_loss), axis=0)

#%%
np.savetxt("training_results.csv", results, delimiter = ",")
print("Validation Accuracy: ", model.evaluate(val_generator))
print("Done!")