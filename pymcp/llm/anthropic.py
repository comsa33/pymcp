import asyncio
import json
from typing import Dict, List, Any, Optional

import anthropic
from pydantic import BaseModel

from pymcp.llm.provider import Provider, Tool, Message, ToolCall, Schema

class ContentBlock(BaseModel):
    """메시지 내용 블록"""
    type: str
    text: Optional[str] = None
    id: Optional[str] = None
    tool_use_id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[bytes] = None
    content: Optional[Any] = None

class AnthropicToolCall(BaseModel):
    """Anthropic 도구 호출 구현"""
    id: str
    name: str
    args: Dict[str, Any]
    
    @property
    def id(self) -> str:
        return self.id
    
    @property
    def name(self) -> str:
        return self.name
    
    def get_arguments(self) -> Dict[str, Any]:
        return self.args

class AnthropicMessage(BaseModel):
    """Anthropic 메시지 구현"""
    id: str
    type: str
    role: str
    content: List[ContentBlock]
    model: str
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Dict[str, int] = {}
    
    @property
    def role(self) -> str:
        return self.role
    
    @property
    def content(self) -> str:
        text_blocks = [block.text for block in self.content if block.type == "text" and block.text]
        return " ".join(text_blocks) if text_blocks else ""
    
    def get_tool_calls(self) -> List[ToolCall]:
        calls = []
        for block in self.content:
            if block.type == "tool_use":
                args = json.loads(block.input) if block.input else {}
                calls.append(AnthropicToolCall(
                    id=block.id or "",
                    name=block.name or "",
                    args=args
                ))
        return calls
    
    def is_tool_response(self) -> bool:
        return any(block.type == "tool_result" for block in self.content)
    
    def get_tool_response_id(self) -> Optional[str]:
        for block in self.content:
            if block.type == "tool_result":
                return block.tool_use_id
        return None
    
    def get_usage(self) -> tuple[int, int]:
        input_tokens = self.usage.get("input_tokens", 0)
        output_tokens = self.usage.get("output_tokens", 0)
        return input_tokens, output_tokens

class AnthropicProvider(Provider):
    """Anthropic Claude 구현"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "claude-3-5-sonnet-20240620"):
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=base_url or None
        )
        self.model = model
    
    async def create_message(
        self, 
        prompt: str, 
        messages: List[Message], 
        tools: List[Tool]
    ) -> Message:
        # Anthropic 메시지 형식으로 변환
        anthropic_messages = []
        
        for msg in messages:
            # 텍스트 콘텐츠 추가
            content = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            
            # 도구 호출 추가
            for call in msg.get_tool_calls():
                content.append({
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.get_arguments()
                })
            
            # 도구 응답 처리
            if msg.is_tool_response():
                tool_id = msg.get_tool_response_id()
                if tool_id:
                    content.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": msg.content
                    })
            
            anthropic_messages.append({
                "role": msg.role,
                "content": content
            })
        
        # 새 프롬프트 추가
        if prompt:
            anthropic_messages.append({
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            })
        
        # Anthropic 도구 형식으로 변환
        anthropic_tools = []
        for tool in tools:
            anthropic_tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": tool.input_schema.type,
                    "properties": tool.input_schema.properties,
                    "required": tool.input_schema.required
                }
            })
        
        # API 호출
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=anthropic_messages,
            tools=anthropic_tools
        )
        
        # 응답을 AnthropicMessage로 변환
        content_blocks = []
        for content_item in response.content:
            if content_item.type == "text":
                content_blocks.append(ContentBlock(
                    type="text",
                    text=content_item.text
                ))
            elif content_item.type == "tool_use":
                content_blocks.append(ContentBlock(
                    type="tool_use",
                    id=content_item.id,
                    name=content_item.name,
                    input=json.dumps(content_item.input).encode()
                ))
        
        return AnthropicMessage(
            id=response.id,
            type=response.type,
            role=response.role,
            content=content_blocks,
            model=response.model,
            stop_reason=response.stop_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
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
            content_str = json.dumps(content)
        
        return AnthropicMessage(
            id="tool_response",
            type="message",
            role="tool",
            content=[ContentBlock(
                type="tool_result",
                tool_use_id=tool_call_id,
                content=content,
                text=content_str
            )],
            model=self.model
        )
    
    def supports_tools(self) -> bool:
        return True
    
    def name(self) -> str:
        return "anthropic"
    