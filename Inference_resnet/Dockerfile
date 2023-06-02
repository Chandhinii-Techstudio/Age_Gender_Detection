FROM ubuntu:latest
WORKDIR /app
# . is for accessing the current directory
#from location(local) . to location(container)
COPY . ./KinetosisInference

RUN apt-get update

RUN apt-get install python3 python3-pip -yq

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install tensorflow

RUN pip install mtcnn

RUN pip install -r ./KinetosisInference/requirements.txt 

CMD ["python3","KinetosisInference/app.py"]