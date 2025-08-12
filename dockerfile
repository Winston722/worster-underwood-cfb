# Use official Python 3.13 image
FROM python:3.13-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy project metadata first (for caching)
COPY pyproject.toml uv.lock* ./

# uv prefers copy mode on container filesystems
ENV UV_LINK_MODE=copy

# Install only runtime deps into project .venv
RUN uv sync --no-dev --frozen

# Make the venv the default runtime Python
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy the rest of the project
COPY . .

# Default command (adjust later if you want to run your job)
CMD ["python", "-c", "import sys; print('container ok:', sys.version)"]

