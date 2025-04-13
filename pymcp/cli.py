import asyncio
import logging
import os
import sys
import re
from typing import Dict, List, Any, Optional

import typer
from dotenv import load_dotenv
from rich.logging import RichHandler

from pymcp.config import Config
from pymcp.llm.provider import Provider, Tool
from pymcp.llm.anthropic import AnthropicProvider
from pymcp.llm.openai import OpenAIProvider
from pymcp.llm.ollama import OllamaProvider
from pymcp.llm.google import GoogleProvider
from pymcp.mcp.client import MCPClient, create_mcp_clients, close_mcp_clients
from pymcp.history import HistoryMessage, add_message_to_history
from pymcp.utils.rendering import (
    render_error, 
    render_tool_call, 
    render_prompt_response, 
    render_tools_list, 
    render_servers_list,
    render_help, 
    render_history
)
from pymcp.utils.terminal import get_user_input, run_with_spinner, print_info

# .env 파일 로드
load_dotenv()

# CLI 앱 생성
app = typer.Typer(
    help="MCPHost - LLM을 MCP를 통해 외부 도구와 통합하는 CLI 호스트 애플리케이션"
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("pymcp")

# 기본 설정
DEFAULT_MODEL = "anthropic:claude-3-5-sonnet-latest"
DEFAULT_MESSAGE_WINDOW = 10

# 전역 상태
provider: Optional[Provider] = None
mcp_clients: Dict[str, MCPClient] = {}
message_history: List[HistoryMessage] = []

async def create_provider(model_string: str, debug: bool = False) -> Provider:
    """모델 문자열에서 제공자 생성"""
    parts = model_string.split(":", 1)
    if len(parts) < 2:
        raise ValueError(f"잘못된 모델 형식. 'provider:model' 형식이 필요합니다. 입력값: {model_string}")
    
    provider_name, model = parts
    
    if debug:
        logger.info(f"제공자 생성: {provider_name}, 모델: {model}")
    
    if provider_name == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API 키가 제공되지 않았습니다. ANTHROPIC_API_KEY 환경 변수를 설정하세요.")
        
        base_url = os.environ.get("ANTHROPIC_API_URL")
        return AnthropicProvider(api_key=api_key, base_url=base_url, model=model)
    
    elif provider_name == "ollama":
        base_url = os.environ.get("OLLAMA_HOST")
        return OllamaProvider(model=model, base_url=base_url)
    
    elif provider_name == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API 키가 제공되지 않았습니다. OPENAI_API_KEY 환경 변수를 설정하세요.")
        
        base_url = os.environ.get("OPENAI_API_URL")
        return OpenAIProvider(api_key=api_key, base_url=base_url, model=model)
    
    elif provider_name == "google":
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Google API 키가 제공되지 않았습니다. GOOGLE_API_KEY 환경 변수를 설정하세요.")
        
        return GoogleProvider(api_key=api_key, model=model)
    
    else:
        raise ValueError(f"지원되지 않는 제공자: {provider_name}")

def mcp_tools_to_llm_tools(server_name: str, mcp_tools: List[Any]) -> List[Tool]:
    """MCP 도구를 LLM 도구로 변환"""
    llm_tools = []
    
    for tool in mcp_tools:
        # 네임스페이스가 있는 도구 이름 생성
        namespaced_name = f"{server_name}__{tool.name}"
        
        llm_tools.append(Tool(
            name=namespaced_name,
            description=tool.description,
            input_schema={
                "type": tool.inputSchema.get("type", "object"),
                "properties": tool.inputSchema.get("properties", {}),
                "required": tool.inputSchema.get("required", [])
            }
        ))
    
    return llm_tools

async def run_prompt(
    prompt: str,
    provider: Provider,
    mcp_clients: Dict[str, MCPClient],
    tools: List[Tool],
    message_history: List[HistoryMessage],
    max_retries: int = 3
) -> None:
    """LLM에 프롬프트 전송 및 응답 처리"""
    # 프롬프트가 있는 경우 표시
    if prompt:
        render_prompt_response("user", prompt)
        
        # 히스토리 생성 및 추가
        user_message = HistoryMessage(
            role="user",
            content=[{"type": "text", "text": prompt}]
        )
        message_history.append(user_message)
    
    retry_count = 0
    backoff = 1.0
    
    while retry_count <= max_retries:
        try:
            # 메시지 생성
            async def create_message_task():
                return await provider.create_message(
                    prompt=prompt if prompt else "",
                    messages=message_history,
                    tools=tools
                )
            
            message = await run_with_spinner("메시지 생성 중...", create_message_task)
            break
        
        except Exception as e:
            error_str = str(e).lower()
            
            # 과부하 오류인 경우 재시도
            if "overloaded" in error_str or "rate limit" in error_str:
                if retry_count >= max_retries:
                    render_error(f"서비스가 과부하 상태입니다. 잠시 후 다시 시도해주세요.")
                    return
                
                logger.warn(f"서비스 과부하, 재시도 중... (시도: {retry_count+1}/{max_retries}, 대기: {backoff}초)")
                await asyncio.sleep(backoff)
                backoff *= 2
                retry_count += 1
                continue
            
            # 다른 오류는 즉시 보고
            render_error(f"메시지 생성 오류: {str(e)}")
            return
    
    # 응답 메시지와 도구 호출 결과 저장
    tool_results = []
    
    # 텍스트 내용이 있으면 표시
    if message.content:
        cleaned_content = re.sub(r'\[TOOL_CALLS\]', '', message.content)
        render_prompt_response("assistant", cleaned_content)
    
    # 메시지 저장
    message_history = add_message_to_history(message_history, message)
    
    # 도구 호출 처리
    for tool_call in message.get_tool_calls():
        # 도구 호출 정보 표시
        render_tool_call(tool_call.name, tool_call.get_arguments())
        
        # 입출력 토큰 사용량 표시
        input_tokens, output_tokens = message.get_usage()
        if input_tokens > 0 or output_tokens > 0:
            logger.info(f"토큰 사용량: 입력={input_tokens}, 출력={output_tokens}, 총={input_tokens+output_tokens}")
        
        # 도구 이름 파싱
        tool_name = tool_call.name
        
        if "__" not in tool_name:
            logger.debug(f"도구 이름에 네임스페이스 없음: {tool_name}, 적절한 서버 찾는 중...")
            
            # 서버별로 도구 확인
            found_server = None
            for server_name, client in mcp_clients.items():
                try:
                    tools_response = await client.list_tools()
                    for tool in tools_response.tools:
                        if tool.name == tool_name:
                            found_server = server_name
                            logger.debug(f"도구 {tool_name}에 적합한 서버 찾음: {server_name}")
                            break
                    if found_server:
                        break
                except Exception as e:
                    logger.debug(f"서버 {server_name}에서 도구 확인 중 오류: {str(e)}")
                    continue
            
            if found_server:
                # 네임스페이스 추가
                qualified_tool_name = f"{found_server}__{tool_name}"
                logger.info(f"도구 이름에 네임스페이스 추가: {tool_name} → {qualified_tool_name}")
                tool_name = qualified_tool_name
            else:
                render_error(f"도구 {tool_name}에 적합한 서버를 찾을 수 없습니다.")
                continue
        
        parts = tool_name.split("__")
        if len(parts) != 2:
            render_error(f"잘못된 도구 이름 형식: {tool_name}")
            continue
        
        server_name, simple_tool_name = parts
        mcp_client = mcp_clients.get(server_name)
        if not mcp_client:
            render_error(f"서버를 찾을 수 없음: {server_name}")
            continue
        
        # 도구 인자 변환
        tool_args = tool_call.get_arguments()
        
        # 도구 호출
        async def call_tool_task():
            return await mcp_client.call_tool(simple_tool_name, tool_args)
        
        try:
            tool_result = await run_with_spinner(f"도구 {simple_tool_name} 실행 중...", call_tool_task)
            
            # 도구 응답 생성
            tool_response = await provider.create_tool_response(
                tool_call_id=tool_call.id,
                content=tool_result.content
            )
            logger.debug(f"도구 {simple_tool_name} 응답: {tool_response.content}")
            
            # 도구 응답 저장
            tool_results.append(tool_response)
            logger.info(f"도구 {simple_tool_name} 응답 저장됨")

            
        except Exception as e:
            error_msg = f"도구 {simple_tool_name} 호출 오류: {str(e)}"
            render_error(error_msg)
            
            # 오류 메시지를 도구 응답으로 저장
            error_response = await provider.create_tool_response(
                tool_call_id=tool_call.id,
                content=error_msg
            )
            tool_results.append(error_response)
    
    # 도구 응답이 있으면 다시 LLM에 요청
    if tool_results:
        # 도구 응답을 히스토리에 추가
        for result in tool_results:
            message_history = add_message_to_history(message_history, result)
        
        # 빈 프롬프트로 재귀 호출
        await run_prompt("", provider, mcp_clients, tools, message_history, max_retries)

async def handle_slash_command(command: str, config: Config, clients: Dict[str, MCPClient], history: List[HistoryMessage]) -> bool:
    """슬래시 명령 처리"""
    if not command.startswith("/"):
        return False
    
    cmd = command.strip().lower()
    
    if cmd == "/help":
        render_help()
        return True
    
    elif cmd == "/tools":
        tools_by_server = {}
        
        async def fetch_tools():
            for server_name, client in clients.items():
                try:
                    tools_response = await client.list_tools()
                    tools_by_server[server_name] = tools_response.tools
                except Exception as e:
                    render_error(f"서버 {server_name}에서 도구 가져오기 실패: {str(e)}")
                    tools_by_server[server_name] = []
        
        await run_with_spinner("모든 서버에서 도구 가져오는 중...", fetch_tools)
        render_tools_list(tools_by_server)
        return True
    
    elif cmd == "/servers":
        render_servers_list(config.get_servers())
        return True
    
    elif cmd == "/history":
        render_history(history)
        return True
    
    elif cmd == "/quit":
        print_info("종료합니다!")
        sys.exit(0)
    
    else:
        render_error(f"알 수 없는 명령: {command}\n명령 목록을 보려면 /help를 입력하세요.")
        return True

@app.command()
def main(
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="사용할 모델 (형식: provider:model)"),
    config_path: Optional[str] = typer.Option(None, "--config", help="설정 파일 경로 (기본값: ~/.mcp.json)"),
    message_window: int = typer.Option(DEFAULT_MESSAGE_WINDOW, "--message-window", help="컨텍스트에 유지할 메시지 수"),
    debug: bool = typer.Option(False, "--debug", help="디버그 로깅 활성화")
):
    """MCPHost - 다양한 LLM을 MCP 서버와 통합하는 CLI 도구"""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("디버그 모드 활성화")
    
    # 비동기 메인 함수 실행
    asyncio.run(async_main(model, config_path, message_window, debug))

async def async_main(model: str, config_path: Optional[str], message_window: int, debug: bool):
    """비동기 메인 함수"""
    global provider, mcp_clients, message_history
    
    try:
        # 설정 로드
        config = Config(config_path=config_path)
        logger.info("설정 로드됨")
        
        # 제공자 생성
        provider = await create_provider(model, debug)
        logger.info(f"모델 로드됨: {model}")
        
        # MCP 클라이언트 생성
        servers_config = config.get_servers()
        if not servers_config:
            logger.warning("MCP 서버가 구성되지 않았습니다. 도구 기능을 사용할 수 없습니다.")
        
        mcp_clients = await create_mcp_clients(servers_config)
        for name in mcp_clients:
            logger.info(f"서버 연결됨: {name}")
        
        # 도구 목록 가져오기
        all_tools = []
        for server_name, client in mcp_clients.items():
            try:
                tools_response = await client.list_tools()
                server_tools = mcp_tools_to_llm_tools(server_name, tools_response.tools)
                all_tools.extend(server_tools)
                logger.info(f"도구 로드됨: 서버={server_name}, 개수={len(tools_response.tools)}")
            except Exception as e:
                logger.error(f"도구 가져오기 실패: 서버={server_name}, 오류={str(e)}")
        
        # 대화 루프
        print_info("MCPHost가 시작되었습니다! 종료하려면 언제든지 /quit 또는 Ctrl+C를 입력하세요.")
        
        while True:
            try:
                # 사용자 입력 받기
                prompt = await get_user_input("메시지를 입력하세요 (명령어 목록을 보려면 /help)")
                
                if not prompt:
                    continue
                
                # 슬래시 명령 처리
                handled = await handle_slash_command(prompt, config, mcp_clients, message_history)
                if handled:
                    continue
                
                # 메시지 윈도우 제한
                if len(message_history) > message_window:
                    message_history = message_history[-message_window:]
                
                # 프롬프트 실행
                await run_prompt(prompt, provider, mcp_clients, all_tools, message_history)
                
            except KeyboardInterrupt:
                print_info("\n종료합니다!")
                break
            except Exception as e:
                render_error(f"오류 발생: {str(e)}")
                if debug:
                    logger.exception("상세 오류:")
    
    finally:
        # 리소스 정리
        if mcp_clients:
            await close_mcp_clients(mcp_clients)
            logger.info("모든 MCP 서버 연결이 종료되었습니다.")

if __name__ == "__main__":
    app()
