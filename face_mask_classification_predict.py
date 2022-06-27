import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import aiplatform as aip
from PIL import Image
from matplotlib import image
import matplotlib.pyplot as plt
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="daring-hash-348101-9717f041dd58.json"
PROJECT_ID = "daring-hash-348101"  # @param {type:"string"}
REGION = "us-east1"  # @param {type: "string"}
BUCKET_NAME = "seangoh-smu-mle-usa"
BUCKET_URI = f"gs://{BUCKET_NAME}"
ENDPOINT = '6005374181238112256'

aip.init(project=PROJECT_ID, location=REGION)

def load_image(image_file):
	img = Image.open(image_file)
	return img

def predict_image(img_path, endpoint):
    """Get predictions for images and plot out with the prediction
    
    Args:
    img_path (str): path to the image
    endpoint (str): endpoint id
    
    Returns:
    matplotlib plot with class prediction 
    """
    
    # convert image for plotting and sending request
    img = np.asarray(Image.open(img_path))
    image_data = np.resize(img, (1,200,200,3)) / 255.0
    image_data_norm = image_data.tolist()
    
    # get prediction
    endpoint = aip.Endpoint(ENDPOINT)
    prediction = endpoint.predict(instances=image_data_norm)
    result = np.argmax(prediction.predictions)
    result = 'With Mask' if result == 1 else 'Without Mask'
    
    # # plot image with prediction
    # plt.imshow(img, interpolation='nearest')
    # plt.suptitle('Prediction = {}'.format(result))
    # plt.axis('off')
    # plt.show()
    return result

def load_image(image_file):
	img = Image.open(image_file)
	return img

st.title('Face Mask Classification')

# image_file = st.file_uploader("Choose a file", type=["png","jpg","jpeg"])
# if image_file is not None:
#     print(image_file)
#      # To See details
#     file_details = {"filename":image_file.name, "filetype":image_file.type,
#                     "filesize":image_file.size}
#     # st.write(file_details)

#     # make prediction from endpoint
#     result = predict_image(image_file, ENDPOINT)

#     # To View Uploaded Image
#     st.image(load_image(image_file),width=250, caption="Prediction: {}".format(result))

    

image_files = st.file_uploader("Choose Files", accept_multiple_files=True, type=["png","jpg","jpeg"])
results = []
loaded_imgs = []
captions = []

for image_file in image_files:
     # make prediction from endpoint
    result = predict_image(image_file, ENDPOINT)
    results.append(result)
    loaded_imgs.append(load_image(image_file))
    captions.append("Prediction: {}".format(result))

st.image(loaded_imgs, width=250, caption=captions)