!git clone https://{username}:{token}@github.com/tituslhy/Face-mask-classification.git

import numpy as np
from matplotlib import image
import sys
sys.path.append('/content/Face-mask-classification')
from src.get_image_brightness import get_image_brightness

from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Colab Notebooks/MLE/

from google.colab import auth
auth.authenticate_user()

#Copy 1 image file from cloud storage bucket for testing.
!gsutil cp gs://seangoh-smu-mle-usa/FaceMask/Test/WithMask/1163.png .


#Unit testing
def test_return_pass_if_float():
	assert get_image_brightness('./1163.png').dtype == 'float32',"output is not float32"

def test_return_pass_if_not_None():
	assert get_image_brightness('./1163.png')!= None,"Output is None"
