FROM python:3.8

RUN pip install pipenv

ENV PYTHONUNBUFFERED=1
ENV PROJECT_DIR /usr/project

WORKDIR ${PROJECT_DIR}

COPY . ${PROJECT_DIR}/

RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pipenv lock
RUN pipenv install --system

COPY ./entrypoint.sh /
ENTRYPOINT [ "sh", "/entrypoint.sh" ]
