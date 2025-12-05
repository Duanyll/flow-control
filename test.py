# nested_progress_demo.py
import itertools
import random
import time

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()

def main():
    # 定义几种不同的进度条样式（你可以自定义）
    def make_progress(style_name: str):
        if style_name == "download":
            return Progress(
                TextColumn("[bold cyan]{task.description}", justify="right"),
                BarColumn(complete_style="cyan", finished_style="bold cyan"),
                "[progress.percentage]{task.percentage:>3.0f}%",
                "•",
                TimeElapsedColumn(),
                "•",
                TimeRemainingColumn(),
                console=console,        # 关键：都使用同一个 console
                transient=False,        # 保持显示，方便看嵌套效果
            )
        elif style_name == "processing":
            return Progress(
                TextColumn("[bold magenta]{task.description}"),
                BarColumn(complete_style="magenta", finished_style="bold magenta"),
                "[progress.percentage]{task.percentage:>3.0f}%",
                console=console,
                transient=True,         # 完成后自动消失
            )
        else:
            return Progress(
                TextColumn("[bold green]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                console=console,
            )

    # 创建多个 Progress 对象
    progress_download = make_progress("download")
    progress_process  = make_progress("processing")
    progress_compress = make_progress("other")

    # 把它们放进一个表格，让它们垂直排列（真正的“嵌套”显示效果）
    # table = Table.grid(expand=True)
    # table.add_row(progress_download)
    # table.add_row(progress_process)
    # table.add_row(progress_compress)

    # 用 with 同时启动所有进度条
    # with console.status("") as status:  # 防止光标跳动
    with progress_download, progress_process, progress_compress:
        # console.print(table)   # 这一行把三条进度条一次性打印出来

        # 示例任务 1：下载 3 个大文件
        download_tasks = [
            progress_download.add_task(f"[cyan]下载 {name}...", total=100)
            for name in ["数据集A.zip", "模型权重.pth", "视频集合.mp4"]
        ]

        # 示例任务 2：处理 10 批数据
        process_task = progress_process.add_task("[magenta]数据预处理", total=10)

        # 示例任务 3：压缩最终结果
        compress_task = progress_compress.add_task("[green]压缩输出文件", total=50)

        # 模拟工作
        for step in range(100):
            time.sleep(0.05 + random.uniform(0, 0.03))  # 模拟网络/IO延迟

            # 更新下载进度（每个文件进度不同）
            for task_id in download_tasks:
                # 让三个文件进度稍微错开
                advance = random.randint(0, 3) if random.random() > 0.3 else 0
                progress_download.update(task_id, advance=advance)

            # 每 10 步处理一批数据
            if step % 10 == 0:
                progress_process.advance(process_task)
                console.log(f"[magenta]已处理 {progress_process.tasks[0].completed} 批数据")

            # 前 50 步在压缩（模拟并行）
            if step < 50:
                progress_compress.advance(compress_task, advance=random.randint(0, 2))

        # 让进度条有时间显示 100% 状态
        time.sleep(0.5)

    console.print("\n[bold green]所有任务完成！[/bold green]")


if __name__ == "__main__":
    # 推荐使用 python -m rich 方式运行，能看到更好的颜色效果
    main()