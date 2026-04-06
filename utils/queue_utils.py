"""
queue_utils.py - 队列操作工具函数
提供：
- get_with_interrupt: 同时等待队列数据和中断信号
- put_with_interrupt: 带中断的队列写入（较少用，但对称提供）
- QueueMonitor: 简单的队列监控（调试用）
"""

import asyncio
from typing import TypeVar, Optional, Any

T = TypeVar('T')


async def get_with_interrupt(
        queue: asyncio.Queue[T],
        interrupt_queue: asyncio.Queue,
        interrupt_value: str = "interrupt"
) -> Optional[T]:
    """
    同时等待从 queue 获取数据，或收到中断信号。

    参数:
        queue: 要读取的数据队列
        interrupt_queue: 中断信号队列
        interrupt_value: 识别中断信号的值（默认为 "interrupt"）

    返回:
        如果正常获取数据，返回数据项；
        如果收到中断信号，返回 None（中断信号被消耗）。
    """
    # 创建两个任务：获取数据和等待中断
    get_task = asyncio.create_task(queue.get())
    interrupt_task = asyncio.create_task(interrupt_queue.get())

    # 等待任一任务完成
    done, pending = await asyncio.wait(
        [get_task, interrupt_task],
        return_when=asyncio.FIRST_COMPLETED
    )

    # 取消尚未完成的任务
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # 判断哪个任务完成了
    if interrupt_task in done:
        # 收到中断信号
        val = interrupt_task.result()
        # 如果信号值匹配（可选检查），返回 None 表示中断
        # 这里简单返回 None，并丢弃中断信号（不再放回）
        return None
    else:
        # 正常获取到数据
        return get_task.result()


async def put_with_interrupt(
        queue: asyncio.Queue[T],
        item: T,
        interrupt_queue: asyncio.Queue,
        interrupt_value: str = "interrupt"
) -> bool:
    """
    尝试向队列放入数据，同时监听中断信号。
    如果放入过程中收到中断，则放弃放入并返回 False。

    返回:
        True 表示成功放入，False 表示被中断。
    """
    put_task = asyncio.create_task(queue.put(item))
    interrupt_task = asyncio.create_task(interrupt_queue.get())

    done, pending = await asyncio.wait(
        [put_task, interrupt_task],
        return_when=asyncio.FIRST_COMPLETED
    )

    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    if interrupt_task in done:
        # 收到中断，放弃放入
        interrupt_task.result()  # 消耗中断信号
        return False
    else:
        # 放入成功
        return True


class QueueMonitor:
    """
    简单的队列监控器，用于记录队列的最大长度和出入次数。
    可用于调试或性能分析。
    """

    def __init__(self, queue: asyncio.Queue, name: str = "Queue"):
        self.queue = queue
        self.name = name
        self.max_size = 0
        self.put_count = 0
        self.get_count = 0

    def record_put(self) -> None:
        """每次放入数据后调用"""
        self.put_count += 1
        self.max_size = max(self.max_size, self.queue.qsize())

    def record_get(self) -> None:
        """每次取出数据后调用"""
        self.get_count += 1

    def report(self) -> str:
        """返回监控摘要"""
        return (f"{self.name}: max_size={self.max_size}, "
                f"puts={self.put_count}, gets={self.get_count}")

    def reset(self) -> None:
        """重置统计数据"""
        self.max_size = 0
        self.put_count = 0
        self.get_count = 0