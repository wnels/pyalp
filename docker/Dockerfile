FROM python:3.10

RUN pip install \
  matplotlib \
  numpy \
  scipy \
  tqdm

RUN useradd --create-home --shell /bin/bash developer

USER developer

WORKDIR /home/developer

COPY --chown=developer . ./pyALP

