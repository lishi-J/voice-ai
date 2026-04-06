"""
modules/audio_output.py - 音频播放模块
从 audio_queue 获取音频块（bytes, 16-bit PCM），通过扬声器播放。
支持中断信号，收到中断时清空播放队列并停止当前播放。
"""

import asyncio
import logging
import sounddevice as sd
import numpy as np
from config import TTS_SAMPLE_RATE

logger = logging.getLogger(__name__)

class AudioOutput:
    def __init__(self, audio_queue, interrupt_queue, sample_rate=TTS_SAMPLE_RATE, channels=1, blocksize=1024):
        """
        :param audio_queue: 输入音频块队列（bytes, 16-bit PCM）
        :param interrupt_queue: 中断信号队列
        :param sample_rate: 音频采样率（应与 TTS 输出一致，edge-tts 默认 24000）
        :param channels: 声道数（通常为 1）
        :param blocksize: 每次写入的块大小（帧数），不影响播放，仅用于底层缓冲
        """
        self.audio_queue = audio_queue
        self.interrupt_queue = interrupt_queue
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize

        # 音频流对象，在 run 中创建
        self.stream = None

    async def run(self):
        """主循环：初始化音频流，不断从队列获取音频块并播放"""
        logger.info("音频输出模块开始运行")

        # 创建输出流（非阻塞模式，使用 int16 格式）
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='int16',
            blocksize=self.blocksize
        )
        self.stream.start()
        logger.info(f"音频输出流已启动，采样率 {self.sample_rate} Hz")

        loop = asyncio.get_event_loop()

        try:
            while True:
                # 同时等待音频块和中断信号
                audio_chunk = await self._get_with_interrupt()
                if audio_chunk is None:
                    # 收到中断：清空播放队列并丢弃当前数据
                    logger.info("音频输出收到中断，清空队列")
                    await self._clear_queue()
                    continue

                # 将 bytes 转换为 numpy int16 数组（保持形状 (n_frames,)）
                audio_np = np.frombuffer(audio_chunk, dtype=np.int16)

                # 如果流已停止（例如被中断后），可能需要重新启动？但一般不会
                # 直接写入流（阻塞，放入线程池执行）
                await loop.run_in_executor(None, self.stream.write, audio_np)
        except asyncio.CancelledError:
            logger.info("音频输出模块被取消")
        finally:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                logger.debug("音频输出流已关闭")

    async def _get_with_interrupt(self):
        """同时等待音频块和中断信号，返回音频块或 None（中断）"""
        get_task = asyncio.create_task(self.audio_queue.get())
        interrupt_task = asyncio.create_task(self.interrupt_queue.get())

        done, pending = await asyncio.wait(
            [get_task, interrupt_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        if interrupt_task in done:
            interrupt_task.result()  # 消耗信号
            return None
        else:
            return get_task.result()

    async def _clear_queue(self):
        """清空播放队列（丢弃所有待播放的音频块）"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        # 可选：立即停止当前播放（但无法停止正在写入的块，只能丢弃后续）
        # 如果需要立即静音，可以调用 self.stream.stop() 再重新 start()
        # 但 stop() 是阻塞的，可能影响实时性。这里简单清空队列即可。
        # 如果希望立刻静音，可以添加：
        # self.stream.stop()
        # self.stream.start()
        # 但注意 stop() 会丢弃缓冲区中尚未播放的数据，可能引起卡顿。
        # 根据需求决定是否实现。