import json
from typing import Dict, List, Any, Optional

import google.generativeai as genai
from pydantic import BaseModel

from pymcp.llm.provider import Provider, Tool, Message, ToolCall, Schema

class GoogleToolCall(BaseModel):
    """Google 도구 호출 구현"""
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

class GoogleMessage(BaseModel):
    """Google 메시지 구현"""
    role: str
    content: str = ""
    tool_calls: List[GoogleToolCall] = []
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
        return self.tool_call_id is not None
    
    def get_tool_response_id(self) -> Optional[str]:
        return self.tool_call_id
    
    def get_usage(self) -> tuple[int, int]:
        # Google doesn't provide token usage info
        return 0, 0

class GoogleProvider(Provider):
    """Google Gemini 모델 구현"""
    
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        genai.configure(api_key=api_key)
        self.model = model
        self.model_client = genai.GenerativeModel(model_name=model)
        self.chat_session = self.model_client.start_chat(history=[])
        self.tool_call_counter = 0
    
    async def create_message(
        self, 
        prompt: str, 
        messages: List[Message], 
        tools: List[Tool]
    ) -> Message:
        # Google 채팅 이력 구성
        history = []
        
        for msg in messages:
            # 도구 호출이 있는 경우
            tool_calls = msg.get_tool_calls()
            if len(tool_calls) > 0:
                for call in tool_calls:
                    history.append({
                        "role": msg.role,
                        "parts": [{
                            "function_call": {
                                "name": call.name,
                                "args": call.get_arguments()
                            }
                        }]
                    })
            
            # 도구 응답인 경우
            elif msg.is_tool_response():
                history.append({
                    "role": "function",
                    "parts": [{
                        "text": msg.content
                    }]
                })
            
            # 일반 텍스트 메시지
            elif msg.content:
                history.append({
                    "role": msg.role,
                    "parts": [{
                        "text": msg.content
                    }]
                })
        
        # Google 도구 형식으로 변환
        google_tools = []
        for tool in tools:
            # 도구 파라미터 변환
            param_schema = self._translate_to_google_schema(tool.input_schema)
            
            google_tools.append({
                "function_declarations": [{
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": param_schema
                }]
            })
        
        # 모델 구성 업데이트
        self.model_client = genai.GenerativeModel(
            model_name=self.model,
            tools=google_tools if tools else None
        )
        
        # 채팅 세션 초기화
        self.chat_session = self.model_client.start_chat(history=history)
        
        # 새 프롬프트 전송
        response = await self.chat_session.send_message_async(prompt if prompt else " ")
        
        # 응답 파싱
        content = ""
        tool_calls = []
        
        # 텍스트 내용 추출
        for part in response.parts:
            if hasattr(part, 'text') and part.text:
                content += part.text
        
        # 도구 호출 추출
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, 'function_call'):
                    fc = part.function_call
                    call_id = f"tool_{self.tool_call_counter}"
                    self.tool_call_counter += 1
                    
                    tool_calls.append(GoogleToolCall(
                        id=call_id,
                        name=fc.name,
                        args=fc.args
                    ))
        
        return GoogleMessage(
            role="assistant",
            content=content,
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
        
        return GoogleMessage(
            role="function",
            content=content_str,
            tool_call_id=tool_call_id
        )
    
    def supports_tools(self) -> bool:
        # Gemini Pro는 도구 지원
        return "pro" in self.model.lower()
    
    def name(self) -> str:
        return "google"
    
    def _translate_to_google_schema(self, schema: Schema) -> Dict[str, Any]:
        """JSON 스키마를 Google 스키마로 변환"""
        result = {
            "type": self._to_google_type(schema.type),
            "required": schema.required or []
        }
        
        # 속성이 있는 경우
        if schema.properties:
            result["properties"] = {}
            for name, prop in schema.properties.items():
                if isinstance(prop, dict):
                    result["properties"][name] = self._property_to_google_schema(prop)
        
        # 속성이 없을 경우 더미 속성 추가
        # Google/Gemini는 빈 객체 스키마를 좋아하지 않음
        if not schema.properties or len(schema.properties) == 0:
            result["properties"] = {
                "unused": {
                    "type": "INTEGER",
                    "nullable": True
                }
            }
        
        return result
    
    def _property_to_google_schema(self, prop: Dict[str, Any]) -> Dict[str, Any]:
        """속성을 Google 스키마로 변환"""
        result = {
            "type": self._to_google_type(prop.get("type", "string"))
        }
        
        if "description" in prop:
            result["description"] = prop["description"]
        
        # 객체인 경우 속성 재귀적으로 변환
        if result["type"] == "OBJECT" and "properties" in prop:
            result["properties"] = {}
            for name, subprop in prop["properties"].items():
                result["properties"][name] = self._property_to_google_schema(subprop)
        
        # 배열인 경우 항목 변환
        elif result["type"] == "ARRAY" and "items" in prop:
            result["items"] = self._property_to_google_schema(prop["items"])
        
        return result
    
    def _to_google_type(self, type_str: str) -> str:
        """JSON 스키마 타입을 Google 타입으로 변환"""
        type_map = {
            "string": "STRING",
            "boolean": "BOOLEAN",
            "object": "OBJECT",
            "array": "ARRAY",
            "integer": "INTEGER",
            "number": "NUMBER"
        }
        return type_map.get(type_str.lower(), "STRING")
