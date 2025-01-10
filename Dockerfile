FROM python:3.10-slim


WORKDIR /app

COPY src/fine-tuning/target/final-model ./final-model/


RUN apt-get update && apt-get install -y \
    python3-dev \
    curl \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir vllm bitsandbytes

RUN curl -fsSL https://ollama.com/install.sh | sh
RUN ollama serve & \
    sleep 5 && \
    ollama pull mxbai-embed-large && \
    pkill ollama

CMD ["sh", "-c", \
    "ollama serve  & \
    vllm serve ./final-model --tokenizer ./final-model --dtype float16 --load_format bitsandbytes --quantization bitsandbytes --max_model_len 4096" \
]