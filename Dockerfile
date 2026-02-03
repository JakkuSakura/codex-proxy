FROM python:3.11-slim

# Create a non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src ./src

# Set python path
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=DEBUG
ENV DEBUG=true

# Entry point

# Expose port
EXPOSE 8765

# Switch to non-root user
USER appuser

# Run the application
CMD ["python", "-m", "codex_proxy"]
