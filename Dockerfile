FROM python:3.8-slim 
WORKDIR /code
COPY requirements.txt .
RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt
COPY src/ .
CMD [ "python", "server.py"]