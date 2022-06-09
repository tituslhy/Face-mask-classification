
import tensorflow as tf
from tensorflow import keras
import numpy as np
#from PIL import Image
import os
from keras import layers

print(os.getcwd())
print(os.listdir())

#Data is downloaded into the container via shellscript. We believe this is more efficient
traindirectory="/app/data/Train_Small"
testdirectory="/app/data/Test_Small"
image_size=224
TrainData=keras.utils.image_dataset_from_directory(traindirectory, class_names=["WithoutMask","WithMask"], image_size=(image_size,image_size))
TestData=keras.utils.image_dataset_from_directory(testdirectory, class_names=["WithoutMask","WithMask"], image_size=(image_size,image_size))

img_augmentation = keras.models.Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

def build_model(num_classes, IMG_SIZE):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = img_augmentation(inputs) #image augmentation within the model. Should this be good practice? Or do we do it inside the map.
    model = keras.applications.EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")
    
    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)
    

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype='float32', name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label

from datetime import datetime
strategy = tf.distribute.MirroredStrategy()

TrainData.map(normalize_img).prefetch(tf.data.AUTOTUNE).batch(64*strategy.num_replicas_in_sync)
TestData.map(normalize_img).prefetch(tf.data.AUTOTUNE).batch(64*strategy.num_replicas_in_sync)
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '500,520')


with strategy.scope():
    model=build_model(2, 224)
model.fit(TrainData,
        epochs=1,
          validation_data=TestData
         , callbacks=[tboard_callback])
model.save("FaceMaskEfficientNetModel")
