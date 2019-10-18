FROM python:3.6-buster

LABEL version="0.0.1-beta"

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y git

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

RUN mkdir /app
WORKDIR /app
# current dir must be project root
COPY . /app

RUN pip install -r ./requirements.txt

# for fix a stupid bug cased by UCloud which always access /usr/bin/python as python bin
RUN ln -s /usr/bin/python3 /usr/bin/python

## Application related part start ##

RUN mkdir /data
RUN mkdir /model

CMD ["bash", "entry.bash"]