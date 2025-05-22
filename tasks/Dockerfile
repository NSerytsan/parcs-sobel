# tasks/Dockerfile

FROM python:3.9-slim

WORKDIR /sobel

COPY . /sobel

RUN pip install --no-cache-dir -r requirements.txt

CMD ["parcs", "serve", "worker.SobelWorker"]