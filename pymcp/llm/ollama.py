import json
import os
from typing import Dict, List, Any, Optional
import time

import httpx
from pydantic import BaseModel

from pymcp.llm.provider import Provider, Tool, Message, ToolCall, Schema

class OllamaToolCall(BaseModel):
    """Ollama 도구 호출 구현"""
    id: str = ""
    name: str = ""
    args: Dict[str, Any] = {}
    
    @property
    def id(self) -> str:
        return self.id or f"call_{int(time.time() * 1000)}"
    
    @property
    def name(self) -> str:
        return self.name
    
    def get_arguments(self) -> Dict[str, Any]:
        return self.args

class OllamaMessage(BaseModel):
    """Ollama 메시지 구현"""
    role: str
    content: str = ""
    tool_calls: List[OllamaToolCall] = []
    tool_call_id: Optional[str] = None
    
    @property
    def role(self) -> str:
        return self.role
    
    @property
    def content(self) -> str:
        return self.content
    
    def get_tool_calls(self) -> List[ToolCall]:
        return self.tool_calls
    
    def is_tool_response(self) -> bool:
        return self.role == "tool" or self.tool_call_id is not None
    
    def get_tool_response_id(self) -> Optional[str]:
        return self.tool_call_id
    
    def get_usage(self) -> tuple[int, int]:
        # Ollama doesn't provide token usage info
        return 0, 0

class OllamaProvider(Provider):
    """Ollama 모델 구현"""
    
    def __init__(self, model: str, base_url: Optional[str] = None):
        self.model = model
        self.base_url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)
    
    async def create_message(
        self, 
        prompt: str, 
        messages: List[Message], 
        tools: List[Tool]
    ) -> Message:
        # Ollama 메시지 형식으로 변환
        ollama_messages = []
        
        for msg in messages:
            # 도구 응답은 role을 "tool"로 설정
            if msg.is_tool_response():
                ollama_msg = {
                    "role": "tool",
                    # content가 프로퍼티가 아닌 경우 get_content() 메서드 호출
                    "content": msg.content if isinstance(msg.content, str) else (
                        msg.get_content() if hasattr(msg, "get_content") else str(msg.content_text) 
                        if hasattr(msg, "content_text") else "")
                }
                ollama_messages.append(ollama_msg)
                continue
            
            # 기본 메시지
            ollama_msg = {
                "role": msg.role,
                # content가 프로퍼티가 아닌 경우 get_content() 메서드 호출
                "content": msg.content if isinstance(msg.content, str) else (
                    msg.get_content() if hasattr(msg, "get_content") else str(msg.content_text)
                    if hasattr(msg, "content_text") else "")
            }
            
            # 도구 호출 추가
            if msg.role == "assistant" and len(msg.get_tool_calls()) > 0:
                tool_calls = []
                for call in msg.get_tool_calls():
                    tool_calls.append({
                        "function": {
                            "name": call.name,
                            "arguments": call.get_arguments()
                        }
                    })
                
                if tool_calls:
                    ollama_msg["tool_calls"] = tool_calls
            
            # 콘텐츠가 있거나 도구 호출이 있는 경우에만 추가
            if msg.content or (msg.get_tool_calls() and len(msg.get_tool_calls()) > 0):
                ollama_messages.append(ollama_msg)
        
        # 새 프롬프트 추가
        if prompt:
            ollama_messages.append({
                "role": "user",
                "content": prompt
            })
        
        # Ollama 도구 형식으로 변환
        ollama_tools = []
        for tool in tools:
            # 도구 속성 변환
            properties = {}
            for name, prop in tool.input_schema.properties.items():
                if isinstance(prop, dict):
                    prop_item = {
                        "type": prop.get("type", "string"),
                        "description": prop.get("description", "")
                    }
                    
                    # enum이 있으면 추가
                    if "enum" in prop:
                        prop_item["enum"] = prop["enum"]
                    
                    properties[name] = prop_item
            
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": tool.input_schema.type,
                        "required": tool.input_schema.required or [],
                        "properties": properties
                    }
                }
            })
        
        # API 호출
        data = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False
        }
        
        if ollama_tools:
            data["tools"] = ollama_tools
        
        response = await self.client.post("/api/chat", json=data)
        response.raise_for_status()
        result = response.json()
        
        # 응답 파싱
        message = result.get("message", {})
        
        # 도구 호출 파싱
        tool_calls = []
        for tool_call in message.get("tool_calls", []):
            function = tool_call.get("function", {})
            tool_calls.append(OllamaToolCall(
                id=tool_call.get("id", f"tc_{int(time.time() * 1000)}"),
                name=function.get("name", ""),
                args=function.get("arguments", {})
            ))
        
        return OllamaMessage(
            role=message.get("role", "assistant"),
            content=message.get("content", ""),
            tool_calls=tool_calls
        )
    
    async def create_tool_response(
        self, 
        tool_call_id: str, 
        content: Any
    ) -> Message:
        # 도구 응답 형식으로 반환
        content_str = ""
        if isinstance(content, str):
            content_str = content
        elif isinstance(content, bytes):
            content_str = content.decode('utf-8')
        else:
            try:
                content_str = json.dumps(content)
            except:
                content_str = str(content)
        
        return OllamaMessage(
            role="tool",
            content=content_str,
            tool_call_id=tool_call_id
        )
    
    def supports_tools(self) -> bool:
        # 모델이 도구를 지원하는지 확인하려면 모델 정보를 가져와야 함
        # 지금은 항상 지원한다고 가정
        return True
    
    def name(self) -> str:
        return "ollama"
    
    async def close(self):
        await self.client.aclose()
