FROM python:3.12-slim

# 비루트 사용자 생성
RUN groupadd -r pymcp && useradd -r -g pymcp pymcp

WORKDIR /app

# 의존성 설치를 위한 기본 도구
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    # MCP 서버 실행에 필요한 nodejs 설치
    nodejs \
    npm \
    # 보안 업데이트 및 정리
    && apt-get upgrade -y \
    && apt-get clean \
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

# 데이터 디렉토리 생성
RUN mkdir -p /app/data \
    && chown -R pymcp:pymcp /app/data

# 비루트 사용자로 전환
USER pymcp

# 헬스체크 엔드포인트 포트 (필요시 사용)
# EXPOSE 5000

# 실행
ENTRYPOINT ["python", "-m", "pymcp.cli"]