FROM python:3.8 as base

RUN pip install pipenv
RUN pip install mysqlclient

ENV PYTHONUNBUFFERED=1
ENV PROJECT_DIR /usr/project

WORKDIR ${PROJECT_DIR}

COPY . ${PROJECT_DIR}/

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN sudo apt-get -y install libcudart11.0
RUN pipenv lock
RUN pipenv install --system --deploy