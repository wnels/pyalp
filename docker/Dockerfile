FROM python:3.10 AS devel

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

FROM devel AS build-pkg

COPY --chown=developer pyalp ./pyalp
COPY --chown=developer LICENSE .
COPY --chown=developer pyproject.toml .
COPY --chown=developer setup.cfg .
COPY --chown=developer README.md .

RUN python -m pip install --upgrade build
RUN python -m build

FROM scratch as pkg
COPY --from=build-pkg /home/developer/dist /

