
import tensorflow as tf
from tensorflow import keras
import numpy as np
#from PIL import Image
import os
from tensorflow.keras import layers
import datetime
#get daily directory name
Today=datetime.datetime.now()
ds=str(Today)[:10]
dailydatadir='gs://seangoh-smu-mle-usa/DailyQC/{}'.format(ds)


#download data
os.system('gsutil -mq cp -r {} /app/'.format(dailydatadir))
#download model
os.system('gsutil cp -r gs://seangoh-smu-mle-usa/Models/FaceMaskEfficientNetModel /app/')

dailydirectory="/app/{}/".format(ds)
image_size=224
ValidationData=keras.utils.image_dataset_from_directory(dailydirectory, class_names=["WithoutMask","WithMask"], image_size=(image_size,image_size))

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = keras.models.load_model('FaceMaskEfficientNetModel')
          
import numpy as np
all_label=[]
all_pred=[]
for data,labels in ValidationData:
    all_label.extend(list(labels))
    probs=model.predict(data)
    preds=np.argmax(probs, axis=1)
    all_pred.extend(list(preds))
          
from sklearn.metrics import f1_score,accuracy_score
f1=float(f1_score(all_pred,all_label))
accuracy=float(accuracy_score(all_pred,all_label))
predict_1=int(np.sum(all_pred))
predict_0=int(np.sum([1 for i in all_pred if i==0]))
label_1=int(np.sum(all_label))
label_0=int(np.sum([1 for i in all_label if i==0]))
predict_1_percent=float(predict_1/(predict_0+predict_1))
predict_0_percent=float(predict_0/(predict_0+predict_1))
label_1_percent=float(label_1/(label_0+label_1))
label_0_percent=float(label_0/(label_0+label_1))

#creat json
#dailyperformance={'accuracy':accuracy, 'f1':f1}
#import json
#with open('dailyperformance.json', 'w') as f:
#    json.dump(dailyperformance, f)
#os.system('gsutil cp dailyperformance.json gs://seangoh-smu-mle-usa/testupload/')
    
#upload bq    
os.system('gcloud auth activate-service-account --key-file=daring-hash-348101-84e938ac4698.json')
print("prediction_finished")
from google.cloud import bigquery
from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file(
    'daring-hash-348101-84e938ac4698.json', scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

client = bigquery.Client(credentials=credentials, project=credentials.project_id,)
dataset_id='facemask'
schematable_id='dailyqc_results'
table_ref=client.dataset(dataset_id).table(schematable_id)
table=client.get_table(table_ref)
row_insert=[(ds,f1,accuracy,predict_1,predict_0,label_1,label_0,predict_1_percent,predict_0_percent,label_1_percent,label_0_percent)]
client.insert_rows(table,row_insert)


