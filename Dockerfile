FROM python:3.9.0-slim-bullseye


WORKDIR /opt/src/

RUN pip install https://download.pytorch.org/whl/cpu/torch-1.11.0%2Bcpu-cp39-cp39-linux_x86_64.whl

COPY requirements.txt requirements.txt

Run pip install -r requirements.txt \
   && rm -rf /root/.cache/pip

COPY . .
