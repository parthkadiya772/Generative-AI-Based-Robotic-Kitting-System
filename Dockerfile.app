FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/generative_kitting \
    KITTING_PROJECT_ROOT=/app \
    ISAACSIM_PATH=/isaac-sim \
    VIRTUAL_ENV=/app/.aikido
    
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv $VIRTUAL_ENV

COPY generative_kitting/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && iconv -f UTF-16 -t UTF-8 /tmp/requirements.txt > /tmp/requirements-utf8.txt \
    && pip install --no-cache-dir -r /tmp/requirements-utf8.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "/app/generative_kitting/ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
