"""
modules/tts.py - 语音合成模块 (本地 Qwen TTS 1.7B)
从 text_queue 接收流式文本，调用本地 Qwen TTS 模型合成语音，
将 PCM 音频数据放入 audio_queue 供播放。
"""

import asyncio
import logging
import torch
import soundfile as sf
import io
import numpy as np
from typing import Optional
from qwen_tts import Qwen3TTSModel
from config import TTS_MODEL, TTS_SAMPLE_RATE, TTS_VOICE, TTS_DEVICE

logger = logging.getLogger(__name__)

class TTS:
    def __init__(
        self,
        text_queue: asyncio.Queue,
        audio_queue: asyncio.Queue,
        interrupt_queue: asyncio.Queue,
        model: str = TTS_MODEL,
        voice: str = TTS_VOICE,
        sample_rate: int = TTS_SAMPLE_RATE,
        device: str = TTS_DEVICE,
    ):
        """
        :param text_queue: 输入文本队列 (str)
        :param audio_queue: 输出 PCM 音频队列 (bytes)
        :param interrupt_queue: 中断信号队列
        :param model: 本地模型名称或路径路径（如 "./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"）
        :param voice: 预设音色或声音描述
        :param sample_rate: 音频采样率 (应与播放模块一致, 24000)
        :param device: 运行设备 ("cuda" 或 "cpu")
        """
        self.text_queue = text_queue
        self.audio_queue = audio_queue
        self.interrupt_queue = interrupt_queue
        self.voice = voice
        self.sample_rate = sample_rate
        self.device = device

        # 加载模型（在初始化时完成，避免重复加载）
        logger.info(f"正在加载{TTS_MODEL}模型")

        self.model = Qwen3TTSModel.from_pretrained(
            TTS_MODEL,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )

        logger.info("Qwen TTS 模型加载完成")

        # 文本缓冲区（用于流式拼接，模型内部会处理切分）
        self.buffer = ""

    async def run(self):
        """主循环：接收文本并触发合成"""
        logger.info("本地 Qwen TTS 模块开始运行")
        try:
            while True:
                # 同时等待文本和中断信号
                text_chunk = await self._get_with_interrupt()
                if text_chunk is None:
                    # 收到中断：清空缓冲区，停止当前合成（下次请求自动开始新会话）
                    logger.debug("TTS 收到中断，清空缓冲区")
                    self.buffer = ""
                    continue

                self.buffer += text_chunk
                await self._synthesize()

        except asyncio.CancelledError:
            logger.info("TTS 模块被取消")

    async def _get_with_interrupt(self):
        """同时等待文本块和中断信号"""
        get_task = asyncio.create_task(self.text_queue.get())
        interrupt_task = asyncio.create_task(self.interrupt_queue.get())

        done, pending = await asyncio.wait(
            [get_task, interrupt_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in pending:
            task.cancel()

        if interrupt_task in done:
            interrupt_task.result()  # 消耗信号
            return None
        return get_task.result()

    async def _synthesize(self):
        """
        使用本地 Qwen TTS 模型合成语音。
        将生成的音频 PCM 数据放入 audio_queue。
        """
        if not self.buffer.strip():
            logger.debug("缓冲区为空，跳过合成")
            return

        text = self.buffer.strip()
        # 清空缓冲区（假设当前文本块已经处理）
        # 注意：模型支持流式输入，但这里简化处理：每次收到文本块就合成一次
        # 更精细的做法是判断句子结束再合成，但 Qwen TTS 内部可能自动处理切分
        self.buffer = ""

        logger.info(f"合成文本: {text[:50]}...")

        try:
            # 将同步模型调用放入线程池执行，避免阻塞事件循环
            loop = asyncio.get_event_loop()
            audio_array = await loop.run_in_executor(
                None,
                self._synthesize_sync,
                text
            )

            if audio_array is None or len(audio_array) == 0:
                logger.info("合成音频为空")
                return

            # 转换为 bytes (PCM int16)
            audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
            await self.audio_queue.put(audio_bytes)
            logger.info(f"音频已放入队列，长度 {len(audio_bytes)} 字节")

        except Exception as e:
            logger.error(f"合成失败: {e}")

    def _synthesize_sync(self, text: str) -> Optional[np.ndarray]:
        """
        同步执行模型推理（将在线程池中运行）
        返回 float32 类型的音频数组，范围 [-1, 1]
        """
        try:
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language="Chinese",
                speaker="Vivian",
            )
            sf.write("output_custom_voice.wav", wavs[0], sr)
            # 解码音频（假设 outputs 包含音频数据）
            # 不同模型输出格式可能不同，请根据实际情况调整

            audio = ""
            # 确保是 float32 类型
            if wavs[0].dtype != np.float32:
                audio = wavs[0].astype(np.float32)

            return audio

        except Exception as e:
            logger.error(f"模型推理失败: {e}")
            return None