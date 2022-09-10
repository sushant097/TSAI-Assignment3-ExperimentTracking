FROM python:3.9-slim


WORKDIR /opt/src/


COPY requirements.txt requirements.txt

Run pip install -r requirements.txt \
   && rm -rf /root/.cache/pip

COPY . .
