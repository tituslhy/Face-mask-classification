#! /bin/sh
gcloud auth activate-service-account --key-file=daring-hash-348101-9717f041dd58.json
gsutil -mq cp -r gs://seangoh-smu-mle-usa/FaceMask /app
python FaceMaskEfficientNet.py
gsutil -mq cp -r /app/FaceMaskEfficientNetModel gs://seangoh-smu-mle-usa/Models/
gsutil -q cp /app/FaceMaskEfficientNetModel.tflite gs://seangoh-smu-mle-usa/Models/
gsutil -q cp /app/EfficientNetPerformanceComparison.json gs://seangoh-smu-mle-usa/Models/
gsutil -mq cp -r /app/logs gs://seangoh-smu-mle-usa/logs/
rm -r /app/FaceMask
