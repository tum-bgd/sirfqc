FROM tensorflow/tensorflow:2.4.1-gpu-jupyter
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
