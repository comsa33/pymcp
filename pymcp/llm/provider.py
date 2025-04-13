from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Protocol, TypeVar, runtime_checkable
import asyncio
from contextlib import AsyncExitStack
import json

from pydantic import BaseModel, Field

@runtime_checkable
class ToolCall(Protocol):
    """도구 호출 인터페이스"""
    
    @property
    def id(self) -> str:
        """도구 호출 ID"""
        ...
    
    @property
    def name(self) -> str:
        """도구 이름"""
        ...
    
    def get_arguments(self) -> Dict[str, Any]:
        """도구 호출 인자"""
        ...

class Message(Protocol):
    """메시지 인터페이스"""
    
    @property
    def role(self) -> str:
        """메시지 발신자 역할 (user, assistant, system)"""
        ...
    
    @property
    def content(self) -> str:
        """메시지 텍스트 내용"""
        ...
    
    def get_tool_calls(self) -> List[ToolCall]:
        """이 메시지에서 발생한 도구 호출"""
        ...
    
    def is_tool_response(self) -> bool:
        """도구 응답 여부"""
        ...
    
    def get_tool_response_id(self) -> Optional[str]:
        """응답 대상 도구 호출 ID"""
        ...
    
    def get_usage(self) -> tuple[int, int]:
        """토큰 사용량 통계 (입력, 출력)"""
        ...

class Schema(BaseModel):
    """도구 입력 파라미터 스키마"""
    type: str
    properties: Dict[str, Any]
    required: List[str] = Field(default_factory=list)

class Tool(BaseModel):
    """도구 정의"""
    name: str
    description: str
    input_schema: Schema

class Provider(ABC):
    """LLM 제공자 공통 인터페이스"""
    
    @abstractmethod
    async def create_message(
        self, 
        prompt: str, 
        messages: List[Message], 
        tools: List[Tool]
    ) -> Message:
        """LLM에 메시지 전송 및 응답 수신"""
        pass
    
    @abstractmethod
    async def create_tool_response(
        self, 
        tool_call_id: str, 
        content: Any
    ) -> Message:
        """도구 응답 메시지 생성"""
        pass
    
    @abstractmethod
    def supports_tools(self) -> bool:
        """도구/함수 호출 지원 여부"""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """제공자 이름"""
        pass
