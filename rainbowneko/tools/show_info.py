from rich.console import Console
from rich.panel import Panel
from rich.text import Text

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