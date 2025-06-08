FROM python:3.13.4-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt