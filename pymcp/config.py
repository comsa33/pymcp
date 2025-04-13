from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import os

from pydantic import BaseModel, Field
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class ServerConfig(BaseModel):
    """MCP 서버 설정"""
    command: str
    args: List[str]
    env: Optional[Dict[str, str]] = None

class MCPConfig(BaseModel):
    """MCP 전체 설정"""
    mcpServers: Dict[str, ServerConfig] = Field(default_factory=dict)

class Config:
    """애플리케이션 설정 관리"""
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.expanduser("~/.mcp.json")
        self.config = self._load_config()
    
    def _load_config(self) -> MCPConfig:
        """설정 파일 로드"""
        config_path = Path(self.config_path)
        
        # 파일이 없으면 기본 설정으로 생성
        if not config_path.exists():
            default_config = MCPConfig()
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                f.write(default_config.model_dump_json(indent=2))
            return default_config
        
        # 기존 파일 로드
        with open(config_path, "r") as f:
            config_data = json.load(f)
        
        return MCPConfig.model_validate(config_data)
    
    def save_config(self) -> None:
        """설정 파일 저장"""
        with open(self.config_path, "w") as f:
            f.write(self.config.model_dump_json(indent=2))
    
    def get_servers(self) -> Dict[str, ServerConfig]:
        """등록된 서버 목록 반환"""
        return self.config.mcpServers
    
    def add_server(self, name: str, command: str, args: List[str], env: Optional[Dict[str, str]] = None) -> None:
        """서버 추가"""
        self.config.mcpServers[name] = ServerConfig(command=command, args=args, env=env)
        self.save_config()
    
    def remove_server(self, name: str) -> None:
        """서버 제거"""
        if name in self.config.mcpServers:
            del self.config.mcpServers[name]
            self.save_config()
