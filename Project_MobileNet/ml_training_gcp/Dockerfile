FROM gcr.io/deeplearning-platform-release/tf-gpu.2-8
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

ENTRYPOINT ["sh", "initialize.sh"]
