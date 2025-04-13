import json
from typing import Dict, List, Any, Optional

import openai
from openai.types.chat import ChatCompletionMessage
from pydantic import BaseModel

from pymcp.llm.provider import Provider, Tool, Message, ToolCall, Schema

class OpenAIToolCall(BaseModel):
    """OpenAI 도구 호출 구현"""
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

class OpenAIMessage(BaseModel):
    """OpenAI 메시지 구현"""
    role: str
    content: Optional[str] = None
    tool_calls: List[OpenAIToolCall] = []
    tool_call_id: Optional[str] = None
    usage: Dict[str, int] = {}
    
    @property
    def role(self) -> str:
        return self.role
    
    @property
    def content(self) -> str:
        return self.content or ""
    
    def get_tool_calls(self) -> List[ToolCall]:
        return self.tool_calls
    
    def is_tool_response(self) -> bool:
        return self.tool_call_id is not None
    
    def get_tool_response_id(self) -> Optional[str]:
        return self.tool_call_id
    
    def get_usage(self) -> tuple[int, int]:
        input_tokens = self.usage.get("prompt_tokens", 0)
        output_tokens = self.usage.get("completion_tokens", 0)
        return input_tokens, output_tokens

class OpenAIProvider(Provider):
    """OpenAI 모델 구현"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "gpt-4"):
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
    
    async def create_message(
        self, 
        prompt: str, 
        messages: List[Message], 
        tools: List[Tool]
    ) -> Message:
        # OpenAI 메시지 형식으로 변환
        openai_messages = []
        
        for msg in messages:
            # 기본 메시지 구조 생성
            openai_message = {
                "role": msg.role
            }
            
            # 도구 호출이 있는 경우 content는 null이어야 함
            if len(msg.get_tool_calls()) > 0:
                openai_message["content"] = None
                tool_calls = []
                
                for call in msg.get_tool_calls():
                    args_json = json.dumps(call.get_arguments())
                    tool_calls.append({
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": args_json
                        }
                    })
                
                openai_message["tool_calls"] = tool_calls
            else:
                openai_message["content"] = msg.content
            
            # 도구 응답인 경우
            if msg.is_tool_response():
                openai_message["tool_call_id"] = msg.get_tool_response_id()
                # OpenAI는 tool 응답에 role을 "tool"로 설정
                openai_message["role"] = "tool"
            
            openai_messages.append(openai_message)
        
        # 새 프롬프트 추가
        if prompt:
            openai_messages.append({
                "role": "user",
                "content": prompt
            })
        
        # OpenAI 도구 형식으로 변환
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": tool.input_schema.type,
                        "properties": tool.input_schema.properties,
                        "required": tool.input_schema.required or []
                    }
                }
            })
        
        # API 호출
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            tools=openai_tools if tools else None,
            max_tokens=4096,
            temperature=0.7
        )
        
        # 첫 번째 선택 가져오기
        choice = response.choices[0]
        message = choice.message
        
        # 도구 호출 변환
        tool_calls = []
        if message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                
                tool_calls.append(OpenAIToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    args=args
                ))
        
        # 응답 생성
        return OpenAIMessage(
            role=message.role,
            content=message.content,
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
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
            try:
                content_str = json.dumps(content)
            except:
                content_str = str(content)
        
        return OpenAIMessage(
            role="tool",
            content=content_str,
            tool_call_id=tool_call_id
        )
    
    def supports_tools(self) -> bool:
        return True
    
    def name(self) -> str:
        return "openai"
