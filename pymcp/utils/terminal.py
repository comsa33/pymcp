import asyncio
import os
import shutil
import sys
from typing import Dict, Any, Optional, List, Callable

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

console = Console()

def get_terminal_size() -> tuple[int, int]:
    """터미널 크기 반환"""
    return shutil.get_terminal_size((80, 24))

def clear_screen() -> None:
    """화면 지우기"""
    os.system('cls' if os.name == 'nt' else 'clear')

async def async_input(prompt: str = "") -> str:
    """비동기 입력 받기"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: Prompt.ask(prompt))

async def get_user_input(prompt: str = "Enter your message") -> str:
    """사용자 입력 받기"""
    try:
        return await async_input(f"[green]{prompt}[/green]")
    except (KeyboardInterrupt, EOFError):
        console.print("\nGoodbye!")
        sys.exit(0)

async def run_with_spinner(message: str, func: Callable) -> Any:
    """스피너와 함께 함수 실행"""
    result = None
    error = None
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        transient=True
    ) as progress:
        task = progress.add_task(message, total=None)
        
        try:
            # 새 태스크에서 함수 실행
            result = await func()
        except Exception as e:
            error = e
        
        # 태스크 완료 처리
        progress.update(task, completed=True, visible=False)
    
    # 오류가 있으면 발생
    if error:
        raise error
    
    return result

def format_code(code: str, language: str = "python") -> None:
    """코드 구문 강조 출력"""
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    console.print(syntax)

def confirm_action(message: str, default: bool = False) -> bool:
    """사용자에게 확인 질문"""
    return Confirm.ask(message, default=default)

def print_error(message: str) -> None:
    """오류 메시지 출력"""
    console.print(f"[bold red]오류:[/bold red] {message}")

def print_warning(message: str) -> None:
    """경고 메시지 출력"""
    console.print(f"[bold yellow]경고:[/bold yellow] {message}")

def print_success(message: str) -> None:
    """성공 메시지 출력"""
    console.print(f"[bold green]성공:[/bold green] {message}")

def print_info(message: str) -> None:
    """정보 메시지 출력"""
    console.print(f"[bold blue]정보:[/bold blue] {message}")
