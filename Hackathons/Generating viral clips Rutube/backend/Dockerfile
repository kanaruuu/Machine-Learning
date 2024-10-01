FROM python:3.11.10

WORKDIR /src

RUN apt-get update && \
    apt install -y \
    ffmpeg

RUN apt install imagemagick
RUN cat /etc/ImageMagick-6/policy.xml | sed 's/none/read,write/g'> /etc/ImageMagick-6/policy.xml

RUN mkdir "tmp"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
