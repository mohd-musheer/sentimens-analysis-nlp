FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ✅ PRE-DOWNLOAD ALL REQUIRED NLTK DATA
RUN python -m nltk.downloader punkt punkt_tab stopwords wordnet

# Copy application
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]
