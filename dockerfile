# Use official Python 3.13 image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy Poetry setup files first for layer caching
COPY pyproject.toml poetry.lock* ./

# Install Poetry and dependencies
RUN pip install --no-cache-dir poetry && \
    uv config virtualenvs.create false && \
    uv install --no-root

# Copy the rest of the project
COPY . .

# Default command (can be updated later)
CMD ["python"]
