FROM python:3.10

RUN \
apt-get update && \
apt-get -y upgrade && \
apt-get install -y \
  python3-tk \
  sudo \
  x11-apps

RUN \
useradd -ms /bin/bash developer && \
usermod -aG sudo developer

RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER developer
WORKDIR /home/developer
COPY --chown=developer ./requirements.txt .

RUN pip install -r requirements.txt
