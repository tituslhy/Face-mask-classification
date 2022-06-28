import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import aiplatform as aip
from PIL import Image
from matplotlib import image
import matplotlib.pyplot as plt
import cv2
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="mlops-vertexai-smu-d5d316b0b0a5.json"
PROJECT_ID = "mlops-vertexai-smu" 
REGION = "us-central1"  # @param {type: "string"}
ENDPOINT = '4589827727267201024'

aip.init(project=PROJECT_ID, location=REGION)

def load_image(image_file):
	img = Image.open(image_file)
	return img

def predict_image(image, endpoint):
    """Get predictions for images and plot out with the prediction
    
    Args:
    image (array): numpy array of image
    endpoint (str): endpoint id
    
    Returns:
    matplotlib plot with class prediction 
    """

    image_data_list = image.tolist()
    
    # get prediction
    endpoint = aip.Endpoint(endpoint)
    prediction = endpoint.predict(instances=image_data_list)
    result = np.argmax(prediction.predictions)
    result = 'With Mask' if result == 1 else 'Without Mask'

    return result


st.title('Face Mask Classification')


image_files = st.file_uploader("Choose Files", accept_multiple_files=True, type=["png","jpg","jpeg"])
results = []
loaded_imgs = []
captions = []

for image_file in image_files:
     # make prediction from endpoint
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB) / 255.0
    img_resize = cv2.resize(img, dsize=(200, 200), interpolation=cv2.COLOR_BGR2RGB)
    img_resize = np.resize(img_resize, (1,200,200,3))
    loaded_imgs.append(img)

    result = predict_image(img_resize, ENDPOINT)
    results.append(result)
    captions.append("Prediction: {}".format(result))

st.image(loaded_imgs, width=250, caption=captions)