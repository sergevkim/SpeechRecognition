FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# Install prerequested
RUN apt-get -y update && \
    apt-get -y install \
        vim \
        htop \
        git \
        wget \
        sudo \
        software-properties-common \
        unzip \
        tmux \
        tree \
        bash-completion

# Install anaconda
RUN . ~/.bashrc && \
    wget https://repo.continuum.io/archive/Anaconda3-2019.10-Linux-x86_64.sh -O ~/Anaconda3.sh && \
    bash ~/Anaconda3.sh -b && \
    echo 'export PATH="/root/anaconda3/bin:$PATH"' >> ~/.bashrc && \
    rm ~/Anaconda3.sh

# Set normal TZ (mostly for logs)
RUN ln -sf /usr/share/zoneinfo/Europe/Moscow /etc/localtime

# PyTorch
RUN . ~/.bashrc && conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

RUN apt-get -y install libsndfile1

# Requirements
#COPY requirements.txt /root/requirements.txt
#RUN . ~/.bashrc && pip install -r /root/requirements.txt

