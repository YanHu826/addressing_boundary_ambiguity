FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/addressing_boundary_ambiguity

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /tmp/requirements.txt

COPY . .

CMD ["bash"]
