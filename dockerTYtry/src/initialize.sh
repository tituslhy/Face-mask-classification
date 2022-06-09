#! /bin/sh
gsutil -mq cp -r gs://face-mask-dataset-smu/ /app/data/
python ./src/FaceMaskEfficientNet.py
gsutil -mq cp -r /app/FaceMaskEfficientNetModel gs://face-mask-dataset-smu/Models/
gsutil -mq cp -r /app/logs gs://face-mask-dataset-smu/logs/
rm -r /app/data
