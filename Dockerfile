FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

# Install SQLite3 >= 3.35.0
RUN wget https://www.sqlite.org/2021/sqlite-autoconf-3350000.tar.gz \
    && tar -xzf sqlite-autoconf-3350000.tar.gz \
    && cd sqlite-autoconf-3350000 \
    && ./configure \
    && make \
    && make install \
    && cd .. \
    && rm -rf sqlite-autoconf-3350000* \
    && ldconfig

# Set environment variables to ensure the new SQLite3 is used
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV PATH=/usr/local/bin:$PATH

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Run Streamlit
CMD ["streamlit", "run", "main.py", "--server.port=8501"]