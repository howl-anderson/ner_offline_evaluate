FROM python:3.6-buster

LABEL version="0.0.1-beta"

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git
RUN apt-get install -y locales locales-all
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US.UTF-8' LC_ALL='en_US.UTF-8'
RUN dpkg-reconfigure locales

RUN mkdir /app
WORKDIR /app
# current dir must be project root
COPY . /app

RUN pip install -r ./requirements.txt

## Application related part start ##

RUN mkdir /data
RUN mkdir /model
RUN mkdir /output

CMD ["bash", "entry.bash"]