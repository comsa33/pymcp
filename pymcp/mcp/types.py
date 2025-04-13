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
    content: Union[List[Dict[str, Any]], List[ContentBlock], str, Any] = Field(default_factory=list)
    
    model_config = {
        "extra": "allow",  # 추가 필드 허용
        "arbitrary_types_allowed": True  # 임의 유형 허용
    }
    
    def __init__(self, **data):
        # 입력 데이터 정규화
        if "content" in data:
            content = data["content"]
            # 문자열 처리
            if isinstance(content, str):
                data["content"] = [{"type": "text", "text": content}]
            # 비어있는 경우
            elif content is None or (isinstance(content, list) and len(content) == 0):
                data["content"] = [{"type": "text", "text": ""}]
            # 리스트가 아닌 경우
            elif not isinstance(content, list):
                data["content"] = [{"type": "text", "text": str(content)}]
        
        super().__init__(**data)
