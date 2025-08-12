# Use official Python 3.13 image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock* ./

# Install dependency tools and project dependencies
RUN pip install --no-cache-dir uv && \
    uv sync --no-dev --frozen

# Copy the rest of the project
COPY . .

# Default command (can be updated later)
CMD ["python"]
