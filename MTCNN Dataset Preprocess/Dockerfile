FROM ubuntu:latest

WORKDIR /app

COPY . ./DatasetKinetosis

RUN apt-get update

RUN apt-get install python3 python3-pip -yq

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r ./DatasetKinetosis/requirements.txt 

RUN pip install mtcnn