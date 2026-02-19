FROM python:3.13-slim

# No .pyc files in the image, real-time log output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Non-root user (Framework Principle 13: Security by Design)
RUN useradd --create-home appuser

# Dependencies first — cached until requirements.txt changes
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Application code
COPY backend/ /app/backend/
COPY prompts/ /app/prompts/
COPY content/ /app/content/
COPY frontend/ /app/frontend/

# Documentation and config template
COPY .env.example FRAMEWORK.md PLATFORM_VISION.md ROADMAP.md PAGRINDAS.md /app/

USER appuser
EXPOSE 8000

# CMD (not ENTRYPOINT) so the team can override:
#   docker run makaronas python -m pytest backend/tests/ -v
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
