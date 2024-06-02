# Base image
FROM runpod/base:0.4.2-cuda11.8.0

ENV HF_HUB_ENABLE_HF_TRANSFER=0

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Cache Models
RUN mkdir models

# Add src files (Worker Template)
ADD src .

ADD models models

CMD python3.11 -u /rp_handler.py
