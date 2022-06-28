
import tensorflow as tf
from tensorflow import keras
import numpy as np
#from PIL import Image
import os
from tensorflow.keras import layers


#Data is downloaded into the container via shellscript. We believe this is more efficient
traindirectory="/app/FaceMask/Train"
testdirectory="/app/FaceMask/Test"
validationdirectory="/app/FaceMask/Validation"
image_size=224
TrainData=keras.utils.image_dataset_from_directory(traindirectory, class_names=["WithoutMask","WithMask"], image_size=(image_size,image_size))
TestData=keras.utils.image_dataset_from_directory(testdirectory, class_names=["WithoutMask","WithMask"], image_size=(image_size,image_size))
ValidationData=keras.utils.image_dataset_from_directory(validationdirectory, class_names=["WithoutMask","WithMask"], image_size=(image_size,image_size))


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
    #x=inputs
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

from datetime import datetime
strategy = tf.distribute.MirroredStrategy()
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
#options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF


#unbatching as keras.utils.image_dataset_from_directory comes with a default batch
TrainData=TrainData.unbatch().with_options(options)
TestData=TestData.unbatch().with_options(options)
ValidationData=ValidationData.unbatch().with_options(options)

#TrainData=TrainData.map(lambda x, y: (img_augmentation(x), y),num_parallel_calls = tf.data.AUTOTUNE)
TrainData=TrainData.prefetch(tf.data.AUTOTUNE).batch(64*strategy.num_replicas_in_sync)
TestData=TestData.prefetch(tf.data.AUTOTUNE).batch(64*strategy.num_replicas_in_sync)
ValidationData=ValidationData.prefetch(tf.data.AUTOTUNE).batch(64*strategy.num_replicas_in_sync)

logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

early_stop= tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=2
)

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '500,520')
with strategy.scope():
    model=build_model(2, 224)
model.fit(TrainData,
        epochs=1,
          validation_data=TestData
         , callbacks=[tboard_callback,early_stop])
model.save("FaceMaskEfficientNetModel")

import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model('/app/FaceMaskEfficientNetModel')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()
with open('FaceMaskEfficientNetModel.tflite', 'wb') as f:
    f.write(quantized_tflite_model)
    
import numpy as np
from sklearn.metrics import f1_score
def evaluate_model(interpreter,model,dataset):
    #interpreter = tflite intepreter
    #model = full model
    dataset=dataset.with_options(options).unbatch().batch(512)
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

  # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    label_digits =[]
    full_model_matches=[]
    full_model_prediction_digits = []
    full_model_label_digits =[]
    for i, batch in enumerate(dataset):
        #Only Validate for 1 batch
        if i==1:
            break
        print("processing batch: "+str(i+1))
        test_images,test_labels=batch
        
        #evaluate main model
        probs=model.predict(test_images)
        full_model_predictions=np.argmax(probs, axis=1)
        matches=list(np.array(full_model_predictions)==np.array(test_labels))
        full_model_matches.extend(matches)
        full_model_prediction_digits.extend(full_model_predictions)
        full_model_label_digits.extend(test_labels)
        
        #evaluate quantized model
        for n,test_image in enumerate(test_images):
            #print('Evaluated on {n} results so far.'.format(n=i))
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
            test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
            interpreter.set_tensor(input_index, test_image)

    # Run inference.
            interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
            output = interpreter.tensor(output_index)
            digit = np.argmax(output()[0])
            prediction_digits.append(digit)
            label_digits.append(test_labels[n])

    print('\n')
  # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_digits)
    label_digits=np.array(label_digits)
    tflite_accuracy = (prediction_digits == label_digits).mean()
    tflite_f1= f1_score(prediction_digits,label_digits)
    full_model_accuracy=sum(full_model_matches)/len(full_model_matches)
    full_model_f1=f1_score(full_model_prediction_digits,full_model_label_digits)
    return tflite_accuracy, full_model_accuracy, tflite_f1,full_model_f1

interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
interpreter.allocate_tensors()
tflite_accuracy, full_model_accuracy, tflite_f1, full_model_f1 = evaluate_model(interpreter,model,ValidationData)
print("The Full Model Accuracy is: "+str(full_model_accuracy)+" and the Quantized Model Accuracy is: "+str(tflite_accuracy))
print("The Full Model F1 is: "+str(full_model_f1)+" and the Quantized Model F1 is: "+str(tflite_f1))
model_performance={'FullModelAccuracy':full_model_accuracy, "QuantizedModelAccuracy":tflite_accuracy, 'AccuracyDifference':tflite_accuracy-full_model_accuracy
                  ,'FullModelF1':full_model_f1, "QuantizedModelF1":tflite_f1, 'F1Difference':tflite_f1-full_model_f1
                  }

import json
with open('EfficientNetPerformanceComparison.json', 'w') as f:
    json.dump(model_performance, f)
