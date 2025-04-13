from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

class Tool(BaseModel):
    """MCP 도구 정의"""
    name: str
    description: str
    inputSchema: Dict[str, Any]

class ListToolsResponse(BaseModel):
    """도구 목록 응답"""
    tools: List[Tool] = Field(default_factory=list)

class CallToolParams(BaseModel):
    """도구 호출 매개변수"""
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

class CallToolRequest(BaseModel):
    """도구 호출 요청"""
    params: Dict[str, Any] = Field(default_factory=dict)

class ContentBlock(BaseModel):
    """콘텐츠 블록"""
    type: str
    text: Optional[str] = None
    
    class Config:
        extra = "allow"  # 추가 필드 허용

class CallToolResult(BaseModel):
    """도구 호출 결과"""
    content: Union[List[ContentBlock], List[Dict[str, Any]], str] = Field(default_factory=list)
