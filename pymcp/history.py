import json
from typing import Dict, List, Any, Optional, Union

from pydantic import BaseModel, Field, model_serializer

from pymcp.llm.provider import Message, ToolCall

class ContentBlock(BaseModel):
    """메시지 내용 블록"""
    type: str
    text: Optional[str] = None
    id: Optional[str] = None
    tool_use_id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[bytes] = None
    content: Optional[Any] = None
    
    # JSON 직렬화 가능하도록 변환 메서드 추가
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        result = {
            "type": self.type
        }
        if self.text:
            result["text"] = self.text
        if self.id:
            result["id"] = self.id
        if self.tool_use_id:
            result["tool_use_id"] = self.tool_use_id
        if self.name:
            result["name"] = self.name
        if self.input:
            try:
                result["input"] = json.loads(self.input)
            except:
                result["input"] = self.input.decode('utf-8', errors='ignore')
        if self.content:
            result["content"] = self.content
        return result

class HistoryToolCall(BaseModel):
    """저장된 도구 호출 구현"""
    id: str
    name: str
    args: bytes
    
    @property
    def id(self) -> str:
        return self.id
    
    @property
    def name(self) -> str:
        return self.name
    
    def get_arguments(self) -> Dict[str, Any]:
        try:
            return json.loads(self.args.decode('utf-8'))
        except:
            return {}

class HistoryMessage(BaseModel):
    """대화 기록 메시지 구현"""
    role: str
    content: List[ContentBlock] = Field(default_factory=list)
    
    @property
    def role(self) -> str:
        return self.role
    
    def get_content_text(self) -> str:
        # 텍스트 콘텐츠 블록 연결
        text_blocks = []
        for block in self.content:
            if block.type == "text" and block.text:
                text_blocks.append(block.text)
        return " ".join(text_blocks)

    def get_content(self) -> str:
        return self.get_content_text()

    @property
    def content_text(self) -> str:
        return self.get_content_text()

    def get_tool_calls(self) -> List[ToolCall]:
        calls = []
        for block in self.content:
            if block.type == "tool_use":
                calls.append(HistoryToolCall(
                    id=block.id or "",
                    name=block.name or "",
                    args=block.input or b"{}"
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
        return 0, 0  # 기록은 사용량을 추적하지 않음

def add_message_to_history(
    history: List[HistoryMessage],
    message: Message,
    max_messages: int = 10
) -> List[HistoryMessage]:
    """메시지를 대화 기록에 추가"""
    content_blocks = []
    
    # 텍스트 콘텐츠 추가
    if message.content:
        content_blocks.append(ContentBlock(
            type="text",
            text=message.content
        ))
    
    # 도구 호출 추가
    for call in message.get_tool_calls():
        input_json = json.dumps(call.get_arguments()).encode('utf-8')
        content_blocks.append(ContentBlock(
            type="tool_use",
            id=call.id,
            name=call.name,
            input=input_json
        ))
    
    # 도구 응답 처리
    if message.is_tool_response():
        tool_id = message.get_tool_response_id()
        if tool_id:
            content_blocks.append(ContentBlock(
                type="tool_result",
                tool_use_id=tool_id,
                text=message.content,
                content=[ContentBlock(type="text", text=message.content)]
            ))
    
    # 새 메시지 생성
    new_message = HistoryMessage(
        role=message.role,
        content=content_blocks
    )
    
    # 기록에 추가
    history.append(new_message)
    
    # 최대 메시지 수 제한
    if len(history) > max_messages:
        history = prune_messages(history, max_messages)
    
    return history

def prune_messages(messages: List[HistoryMessage], max_messages: int) -> List[HistoryMessage]:
    """오래된 메시지 정리 및 고아 도구 호출/응답 처리"""
    if len(messages) <= max_messages:
        return messages
    
    # 최신 메시지만 유지
    messages = messages[-max_messages:]
    
    # 도구 호출 ID와 응답 ID 수집
    tool_use_ids = set()
    tool_result_ids = set()
    
    for msg in messages:
        for block in msg.content:
            if block.type == "tool_use":
                tool_use_ids.add(block.id)
            elif block.type == "tool_result":
                tool_result_ids.add(block.tool_use_id)
    
    # 고아 도구 호출/응답 필터링
    pruned_messages = []
    for msg in messages:
        pruned_blocks = []
        for block in msg.content:
            keep = True
            if block.type == "tool_use":
                keep = block.id in tool_result_ids
            elif block.type == "tool_result":
                keep = block.tool_use_id in tool_use_ids
            
            if keep:
                pruned_blocks.append(block)
        
        # 보존할 콘텐츠가 있거나 어시스턴트 메시지가 아닌 경우에만 포함
        if pruned_blocks or msg.role != "assistant":
            msg.content = pruned_blocks
            pruned_messages.append(msg)
    
    return pruned_messages

def get_message_text(message: Message) -> str:
    """메시지에서 텍스트 콘텐츠 추출"""
    return message.content or ""
