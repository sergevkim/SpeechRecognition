FROM ubuntu:16.04

# Install prerequested
RUN apt-get -y update && \
    apt-get -y install \
        bash-completion \
        git \
        htop \
        sudo \
        software-properties-common \
        tmux \
        tree \
        unzip \
        vim \
        wget

# Set normal TZ (mostly for logs)
RUN ln -sf /usr/share/zoneinfo/Europe/Moscow /etc/localtime

RUN apt-get -y install libsndfile1

# Requirements
COPY requirements.txt /root/requirements.txt
RUN . ~/.bashrc && pip install -r /root/requirements.txt

