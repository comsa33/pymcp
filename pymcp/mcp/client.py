import os
import logging
from typing import Dict, List, Any, Optional

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from pydantic import BaseModel

from pymcp.mcp.types import Tool, ListToolsResponse, CallToolResult

class MCPClient:
    """MCP 클라이언트"""
    
    def __init__(self, name: str, command: str, args: List[str], env: Optional[Dict[str, str]] = None):
        self.name = name
        self.command = command
        self.args = args
        self.env = env or {}
        self.session = None
        self.exit_stack = None
        self.stdio = None
        self.write = None
        self.logger = logging.getLogger(f"mcp.client.{name}")
    
    async def initialize(self) -> None:
        """클라이언트 초기화 및 서버 연결"""
        import contextlib
        self.exit_stack = contextlib.AsyncExitStack()
        
        # 환경 변수 구성
        env_dict = os.environ.copy()
        if self.env:
            env_dict.update(self.env)
        
        # 서버 매개변수 설정
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=env_dict
        )
        
        # stdio 전송 설정
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            
            # 세션 초기화
            await self.session.initialize()
            self.logger.info(f"MCP 서버 '{self.name}' 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"MCP 서버 '{self.name}' 초기화 실패: {str(e)}")
            await self.close()
            raise
    
    async def list_tools(self) -> ListToolsResponse:
        """서버에서 사용 가능한 도구 목록 요청"""
        if not self.session:
            raise ValueError("세션이 초기화되지 않았습니다")
        
        response = await self.session.list_tools()
        tools = []
        
        for tool in response.tools:
            tools.append(Tool(
                name=tool.name,
                description=tool.description,
                inputSchema={
                    "type": tool.inputSchema.get("type", "object"),
                    "properties": tool.inputSchema.get("properties", {}),
                    "required": tool.inputSchema.get("required", [])
                }
            ))
        
        return ListToolsResponse(tools=tools)
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """도구 호출"""
        if not self.session:
            raise ValueError("세션이 초기화되지 않았습니다")
        
        try:
            # 도구 호출 전에 인자 유효성 검사
            # 인자가 None이거나 빈 객체인 경우 빈 딕셔너리로 설정
            if arguments is None:
                arguments = {}
            
            # 도구 정보 조회 시도
            try:
                tools_list = await self.list_tools()
                tool_found = False
                expected_params = {}
                
                for tool in tools_list.tools:
                    if tool.name == name:
                        tool_found = True
                        expected_params = tool.inputSchema.get("properties", {})
                        break
                
                if not tool_found:
                    self.logger.warning(f"도구 '{name}'을(를) 서버에서 찾을 수 없습니다.")
                    # 도구를 찾을 수 없더라도 호출은 계속 진행
            except Exception as e:
                self.logger.warning(f"도구 정보 조회 중 오류 발생: {str(e)}")
                # 도구 정보를 가져올 수 없더라도 호출은 계속 진행
            
            # 도구 호출
            result = await self.session.call_tool(name, arguments)
            
            # 응답 형식 처리 개선
            try:
                # 1. 응답이 None인 경우
                if result.content is None:
                    return CallToolResult(content=[{"type": "text", "text": "작업이 완료되었습니다."}])
                
                # 2. 문자열인 경우
                if isinstance(result.content, str):
                    return CallToolResult(content=[{"type": "text", "text": result.content}])
                
                # 3. 이미 리스트인 경우
                elif isinstance(result.content, list):
                    content = result.content
                    
                    # 리스트가 비어있는 경우
                    if len(content) == 0:
                        return CallToolResult(content=[{"type": "text", "text": "작업이 완료되었습니다."}])
                    
                    # 각 항목이 딕셔너리가 아니면 변환
                    normalized_content = []
                    for item in content:
                        if isinstance(item, dict) and "type" in item:
                            normalized_content.append(item)
                        else:
                            normalized_content.append({"type": "text", "text": str(item)})
                    
                    return CallToolResult(content=normalized_content)
                
                # 4. 그 외 다른 형식 (JSON 직렬화 시도)
                else:
                    try:
                        import json
                        json_str = json.dumps(result.content, ensure_ascii=False, indent=2)
                        return CallToolResult(content=[
                            {"type": "text", "text": "결과:"},
                            {"type": "code", "text": json_str, "language": "json"}
                        ])
                    except (TypeError, json.JSONDecodeError):
                        # JSON 직렬화 실패 시 문자열로 변환
                        return CallToolResult(content=[{"type": "text", "text": str(result.content)}])
                    
            except Exception as e:
                self.logger.error(f"결과 형식 변환 중 오류: {str(e)}")
                # 오류 발생 시 안전한 응답으로 변환
                return CallToolResult(content=[
                    {"type": "text", "text": f"(응답 처리 오류) 원본 응답: {str(result.content)}"}
                ])
                
        except Exception as e:
            self.logger.error(f"도구 호출 실패 '{name}': {str(e)}")
            # 상세한 오류 메시지 생성
            error_message = f"도구 '{name}' 호출 중 오류 발생:\n{str(e)}"
            
            if hasattr(e, "__traceback__"):
                import traceback
                tb_str = "".join(traceback.format_tb(e.__traceback__))
                self.logger.debug(f"도구 호출 오류 상세 내용:\n{tb_str}")
            
            # 오류 응답 반환
            return CallToolResult(content=[
                {"type": "text", "text": error_message}
            ])
    
    async def close(self) -> None:
        """클라이언트 리소스 정리"""
        if self.exit_stack:
            try:
                await self.exit_stack.aclose()
                self.logger.info(f"MCP 서버 '{self.name}' 연결 종료")
            except Exception as e:
                self.logger.error(f"MCP 서버 '{self.name}' 연결 종료 중 오류: {str(e)}")
            
            self.exit_stack = None
            self.session = None
            self.stdio = None
            self.write = None

async def create_mcp_clients(config: Dict[str, Any]) -> Dict[str, MCPClient]:
    """MCP 서버 설정에서 클라이언트 생성"""
    clients = {}
    initialized = []
    error_messages = []
    
    try:
        for name, server_config in config.items():
            try:
                client = MCPClient(
                    name=name,
                    command=server_config.command,
                    args=server_config.args,
                    env=server_config.env
                )
                
                await client.initialize()
                clients[name] = client
                initialized.append(name)
                
            except Exception as e:
                # 개별 서버 오류는 기록하고 계속 진행
                error_msg = f"서버 '{name}' 초기화 실패: {str(e)}"
                error_messages.append(error_msg)
                logging.error(error_msg)
                
                # 이미 일부 초기화된 경우 리소스 정리
                if hasattr(client, 'close') and callable(client.close):
                    await client.close()
                
                continue
        
        # 모든 서버가 실패하고 하나도 초기화되지 않은 경우
        if not clients and error_messages:
            all_errors = "\n".join(error_messages)
            raise RuntimeError(f"모든 MCP 서버 초기화 실패:\n{all_errors}")
            
    except Exception as e:
        if not isinstance(e, RuntimeError) or "모든 MCP 서버 초기화 실패" not in str(e):
            # 이미 초기화된 클라이언트 정리
            for name in initialized:
                await clients[name].close()
            
            raise RuntimeError(f"MCP 클라이언트 생성 중 오류 발생: {str(e)}")
        else:
            # 이미 생성된 RuntimeError는 그대로 전달
            raise
    
    return clients

async def close_mcp_clients(clients: Dict[str, MCPClient]) -> None:
    """모든 MCP 클라이언트 정리"""
    for name, client in clients.items():
        await client.close()
