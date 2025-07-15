from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.style import Style
from rich.box import ROUNDED

show_set = set()

def show_check_info(title, info, other=None):
    if title not in show_set:
        show_set.add(title)

        console = Console()
        # 创建简洁的状态信息
        text = Text()
        text.append(info+" "*10, style="yellow")
        if other is not None:
            text.append(other)

        # 使用简洁的面板展示
        panel = Panel(
            text,
            title=title,
            border_style="blue",
            expand=False,
            padding=(0, 2)
        )

        # 输出到控制台
        console.print(panel)

def show_note_info(title, info, other=None, once=False):
    if once:
        if title in show_set:
            return
        else:
            show_set.add(title)

    console = Console()
    text = Text()

    # 主信息：明亮柔和的粉色
    text.append(info + " " * 6, style="bold pink1")

    # 附加信息：优雅淡紫色
    if other is not None:
        text.append("\n" + other, style="italic plum1")

    # 创建梦幻粉紫风格面板
    panel = Panel(
        text,
        title=f"[bold medium_violet_red]{title}[/bold medium_violet_red]",
        border_style="orchid",
        box=ROUNDED,
        padding=(0, 2),
        expand=False,
    )

    console.print(panel)

if __name__ == "__main__":
    show_note_info("NekoDataLoader", "'spawn' context is not available, using 'fork' context instead. Please add environment variable 'OMP_NUM_THREADS=1' for better compatibility.", once=True)