#Dockerfile
FROM nvcr.io/nvidia/pytorch:25.05-py3

RUN pip install xformers

RUN cd /opt && \
  git clone https://github.com/irlyngaas/UCF-VIT.git && \
  cd /opt/UCF-VIT && \
  pip install -e .
