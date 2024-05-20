# Base image
FROM runpod/base:0.4.2-cuda11.8.0

ARG CIVITAI_MODEL_ID
ARG CIVITAI_KEY

ENV HF_HUB_ENABLE_HF_TRANSFER=0

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Cache Models
COPY builder/cache_models.py /cache_models.py
RUN mkdir models
RUN python3.11 /cache_models.py --civitai_key $CIVITAI_KEY --model_id $CIVITAI_MODEL_ID && \
    rm /cache_models.py

# Add src files (Worker Template)
ADD src .

ADD models models

CMD python3.11 -u /rp_handler.py
