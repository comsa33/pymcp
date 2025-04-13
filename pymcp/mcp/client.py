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
            result = await self.session.call_tool(name, arguments)
            
            # 응답 형식 처리 개선
            try:
                # 1. 문자열인 경우
                if isinstance(result.content, str):
                    content = [{"type": "text", "text": result.content}]
                # 2. 이미 리스트인 경우
                elif isinstance(result.content, list):
                    content = result.content
                    # 각 항목이 딕셔너리가 아니면 변환
                    for i, item in enumerate(content):
                        if not isinstance(item, dict):
                            content[i] = {"type": "text", "text": str(item)}
                # 3. 그 외 다른 형식
                else:
                    content = [{"type": "text", "text": str(result.content)}]
                    
                return CallToolResult(content=content)
            except Exception as e:
                self.logger.error(f"결과 형식 변환 중 오류: {str(e)}")
                # 안전한 형식으로 변환
                return CallToolResult(content=[{"type": "text", "text": f"(형식 오류) {str(result.content)}"}])
                
        except Exception as e:
            self.logger.error(f"도구 호출 실패 '{name}': {str(e)}")
            raise
    
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
    
    try:
        for name, server_config in config.items():
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
        # 오류 발생 시 이미 초기화된 클라이언트 정리
        for name in initialized:
            await clients[name].close()
        
        raise RuntimeError(f"MCP 클라이언트 생성 실패: {str(e)}")
    
    return clients

async def close_mcp_clients(clients: Dict[str, MCPClient]) -> None:
    """모든 MCP 클라이언트 정리"""
    for name, client in clients.items():
        await client.close()
