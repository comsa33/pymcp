import shutil
import json
from typing import List, Dict, Any

from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

# 콘솔 객체 생성
console = Console()

def get_terminal_width() -> int:
    """터미널 너비 반환"""
    term_width, _ = shutil.get_terminal_size((80, 24))
    return term_width - 4  # 여백용으로 약간 줄임

def render_markdown(text: str) -> None:
    """마크다운 렌더링"""
    md = Markdown(text, code_theme="monokai")
    console.print(md)

def render_error(message: str) -> None:
    """오류 메시지 렌더링"""
    error_panel = Panel(
        Text(message, style="bold red"),
        border_style="red",
        title="오류",
        expand=False
    )
    console.print(error_panel)

def render_tool_call(name: str, args: Dict[str, Any]) -> None:
    """도구 호출 정보 렌더링"""
    console.print(f"[cyan]🔧 도구 호출:[/cyan] [bold]{name}[/bold]")
    
    # 인자가 있으면 테이블로 표시
    if args:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("매개변수", style="cyan")
        table.add_column("값")
        
        for key, value in args.items():
            table.add_row(key, str(value))
        
        console.print(table)

def render_code(code: str, language: str = "python") -> None:
    """코드 구문 강조 렌더링"""
    syntax = Syntax(code, language, theme="monokai", word_wrap=True)
    console.print(syntax)

def render_prompt_response(role: str, content: str) -> None:
    """프롬프트/응답 메시지 렌더링"""
    style = "green" if role == "user" else "blue"
    role_text = "You" if role == "user" else "Assistant"
    
    header = Text(f"{role_text}: ", style=f"bold {style}")
    console.print(header, end="")
    
    # 마크다운 렌더링
    md = Markdown(content)
    console.print(md)

def render_tools_list(tools_by_server: Dict[str, List[Any]]) -> None:
    """서버별 도구 목록 렌더링"""
    for server_name, tools in tools_by_server.items():
        console.print(f"\n[bold purple]서버: {server_name}[/bold purple]")
        
        if not tools:
            console.print("  이용 가능한 도구 없음")
            continue
        
        table = Table(show_header=True, header_style="bold")
        table.add_column("도구", style="cyan")
        table.add_column("설명")
        
        for tool in tools:
            table.add_row(tool.name, tool.description)
        
        console.print(table)

def render_servers_list(servers: Dict[str, Any]) -> None:
    """서버 목록 렌더링"""
    if not servers:
        console.print("[yellow]구성된 서버 없음[/yellow]")
        return
    
    for name, config in servers.items():
        console.print(f"\n[bold purple]서버: {name}[/bold purple]")
        console.print(f"명령어: [cyan]{config.command}[/cyan]")
        
        if config.args:
            console.print("인자:")
            for arg in config.args:
                console.print(f"  - {arg}")
        
        if config.env:
            console.print("환경 변수:")
            for key, value in config.env.items():
                console.print(f"  - {key}: {value}")

def render_help() -> None:
    """도움말 렌더링"""
    help_text = """
# 사용 가능한 명령

다음 명령들을 사용할 수 있습니다:

## 기본 명령
- **/help**: 이 도움말 표시
- **/tools**: 사용 가능한 모든 도구 목록 표시
- **/servers**: 구성된 MCP 서버 목록 표시
- **/history**: 대화 기록 표시
- **/quit**: 애플리케이션 종료

## 서버 관리
- **/addserver <이름> <명령어> [인자...]**: 새 MCP 서버 추가
- **/removeserver <이름>**: 기존 MCP 서버 제거
- **/testserver <이름>**: MCP 서버 연결 테스트 및 재연결

언제든지 Ctrl+C를 눌러 종료할 수 있습니다.

## 사용 가능한 모델

--model 또는 -m 플래그를 사용하여 모델 지정:

- **Anthropic Claude**: `anthropic:claude-3-5-sonnet-latest`
- **OpenAI**: `openai:gpt-4`
- **Ollama 모델**: `ollama:modelname`
- **Google**: `google:gemini-pro`

예시:
```
pymcp -m anthropic:claude-3-5-sonnet-latest
pymcp -m ollama:llama3
```

## 서버 관리 예시

새 MCP 서버 추가:
```
/addserver myserver npx -y @smithery/cli@latest run @myorg/mcp-server
```

서버 연결 테스트:
```
/testserver myserver
```

서버 제거:
```
/removeserver myserver
```
"""
    render_markdown(help_text)

def render_history(messages: List[Any]) -> None:
    """대화 기록 렌더링"""
    console.print("\n[bold]대화 기록[/bold]\n")
    
    for msg in messages:
        role_style = "green" if msg.role == "user" else "blue"
        role_text = "You" if msg.role == "user" else "Assistant"
        
        console.print(f"[bold {role_style}]{role_text}:[/bold {role_style}]")
        
        for block in msg.content:
            if block.type == "text" and block.text:
                console.print(Markdown(block.text))
            
            elif block.type == "tool_use":
                console.print(f"[cyan]🔧 도구 호출:[/cyan] [bold]{block.name}[/bold]")
                if block.input:
                    try:
                        args = json.loads(block.input)
                        for key, value in args.items():
                            console.print(f"  - {key}: {value}")
                    except:
                        pass
            
            elif block.type == "tool_result":
                console.print(f"[yellow]🔄 도구 결과:[/yellow] [dim]({block.tool_use_id})[/dim]")
                if block.text:
                    console.print(Markdown(block.text))
        
        console.print("---")
