FROM python:3.12-slim

WORKDIR /app

# 의존성 설치를 위한 기본 도구
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Poetry 설치
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# 프로젝트 파일 복사
COPY pyproject.toml poetry.lock* ./

# 의존성 설치 (개발 의존성 제외)
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# 소스 코드 복사
COPY pymcp ./pymcp

# 실행
ENTRYPOINT ["python", "-m", "pymcp.cli"]