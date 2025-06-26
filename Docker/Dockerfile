#Dockerfile
FROM nvcr.io/nvidia/pytorch:25.05-py3

RUN pip install xformers \
  timm \
  monai \
  nibabel \
  torchdata==0.9.0 \
  einops \
  opencv-python-headless \
  matplotlib \
  scipy \
  deephyper==0.7.0 \

RUN cd /opt && \
  git clone https://github.com/irlyngaas/UCF-VIT.git && \
  cd /opt/UCF-VIT && \
  pip install -e .
