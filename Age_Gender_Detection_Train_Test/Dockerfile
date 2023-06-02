FROM ubuntu:latest
WORKDIR /app
# . is for accessing the current directory
#from location(local) . to location(container)
COPY . ./kinetosis

RUN apt-get update

RUN apt-get install python3 python3-pip -yq

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r ./kinetosis/requirements.txt 
