FROM python:3.8.1

LABEL maintainer="ruben.nunez@stud.hslu.ch"

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . /app

RUN pip install -r requirements.txt
ENV FLASK_APP=main.py
ENV FLASK_ENV=development

EXPOSE 5000
CMD gunicorn app:app
