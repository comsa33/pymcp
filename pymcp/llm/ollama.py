import json
import os
import re
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
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)
    
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
                # 도구 호출 ID 추가 (Ollama가 지원하는 경우)
                tool_id = msg.get_tool_response_id()
                if tool_id:
                    ollama_msg["tool_call_id"] = tool_id
                    
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
                    # 인자를 직접 객체로 전달 (문자열화하지 않음)
                    tool_calls.append({
                        "id": call.id,
                        "function": {
                            "name": call.name,
                            "arguments": call.get_arguments()  # 직접 객체로 전달
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
            "stream": False,
            "options": {
                "num_predict": 4096,
                "temperature": 0.7
            }
        }
        
        if ollama_tools:
            data["tools"] = ollama_tools

        # 디버그 로깅 - 문제를 진단하기 위한 요청 페이로드 출력 (선택 사항)
        import logging
        logging.debug(f"Ollama API 요청 페이로드: {json.dumps(data, default=str, ensure_ascii=False)}")
        
        try:
            response = await self.client.post("/api/chat", json=data, timeout=120.0)
            response.raise_for_status()
            result = response.json()
            
            # 응답 파싱
            message = result.get("message", {})
            
            # 도구 호출 파싱
            tool_calls = []
            for tool_call in message.get("tool_calls", []):
                if not isinstance(tool_call, dict):
                    continue
                    
                function = tool_call.get("function", {})
                
                # 인자가 문자열로 오는 경우 JSON 파싱 시도
                args = function.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"text": args}
                
                tool_calls.append(OllamaToolCall(
                    id=tool_call.get("id", f"tc_{int(time.time() * 1000)}"),
                    name=function.get("name", ""),
                    args=args
                ))

            content = message.get("content", "")
            
            # 텍스트 형식의 도구 호출 감지 (대체 방법)
            if not tool_calls:
                # 도구 호출 패턴 확인
                patterns = [
                    # 표준 패턴
                    r'([a-zA-Z0-9_-]+)__([a-zA-Z0-9_-]+)\s*(\{.*?\})',
                    # 추가 패턴
                    r'도구 호출:\s*([a-zA-Z0-9_-]+)__([a-zA-Z0-9_-]+)\s*(\{.*?\})',
                    r'\[도구\]\s*([a-zA-Z0-9_-]+)__([a-zA-Z0-9_-]+)\s*(\{.*?\})'
                ]
                
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.DOTALL)
                    for match in matches:
                        try:
                            if len(match.groups()) >= 3:
                                server_name = match.group(1)
                                tool_name = match.group(2)
                                args_str = match.group(3)
                                
                                # 인자 파싱 시도
                                try:
                                    args = json.loads(args_str)
                                except json.JSONDecodeError:
                                    args = {"raw": args_str}
                                
                                # 도구 호출 생성
                                tool_calls.append(OllamaToolCall(
                                    id=f"extracted_{int(time.time() * 1000)}",
                                    name=f"{server_name}__{tool_name}",
                                    args=args
                                ))
                                
                                # 매칭된 부분 제거
                                content = content.replace(match.group(0), "")
                        except Exception as e:
                            import logging
                            logging.error(f"도구 호출 패턴 파싱 오류: {str(e)}")
                
                # TOOL_CALLS 마커 제거
                content = re.sub(r'\[TOOL_CALLS\].*?(\n|$)', '', content, flags=re.DOTALL)
                content = content.strip()
            
            return OllamaMessage(
                role=message.get("role", "assistant"),
                content=content,
                tool_calls=tool_calls
            )
            
        except httpx.TimeoutException:
            # 타임아웃 오류 처리
            raise RuntimeError("Ollama 서버 응답 시간 초과. 서버 부하를 확인하거나 다시 시도하세요.")
        except httpx.HTTPStatusError as e:
            # HTTP 상태 오류 처리
            status_code = e.response.status_code
            error_message = f"Ollama API 오류 (상태 코드: {status_code})"
            
            # 응답 본문에서 자세한 오류 메시지 추출 시도
            try:
                error_json = e.response.json()
                if "error" in error_json:
                    error_message += f": {error_json['error']}"
            except Exception:
                # 응답 본문을 직접 확인
                try:
                    error_message += f" 응답: {e.response.text}"
                except:
                    pass
                
            raise RuntimeError(error_message)
        except Exception as e:
            # 그 외 모든 오류
            raise RuntimeError(f"Ollama 연결 오류: {str(e)}")
    
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
    
    def __del__(self):
        """소멸자에서 비동기 리소스 정리 경고"""
        if hasattr(self, 'client') and not self.client.is_closed:
            import warnings
            warnings.warn("OllamaProvider 리소스가 정리되지 않았습니다. close() 메서드를 명시적으로 호출하세요.")
    
    async def close(self):
        """리소스 정리"""
        if hasattr(self, 'client') and not self.client.is_closed:
            await self.client.aclose()
