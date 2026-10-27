FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by LightGBM and pyarrow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# DB_PATH is overridden at runtime to point at the Render persistent disk.
# SEASON_PHASE defaults to POSTSEASON; set via Render env var.
ENV DB_PATH=/data/mlb_data.db
ENV SEASON_PHASE=POSTSEASON

EXPOSE 8010

COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
