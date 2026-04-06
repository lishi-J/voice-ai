"""
modules/tts.py - 语音合成模块
从 text_queue 接收文本流，按标点切分成句子，调用 edge_tts 合成 MP3，解码为 PCM 后放入 audio_queue。
支持中断信号，收到中断时清空缓冲区并取消当前合成。
"""

import asyncio
import logging
import re
import io
import edge_tts
from pydub import AudioSegment
from config import TTS_VOICE, TTS_SAMPLE_RATE

logger = logging.getLogger(__name__)

class TTS:
    def __init__(self, text_queue, audio_queue, interrupt_queue, voice=TTS_VOICE, sample_rate=TTS_SAMPLE_RATE):
        """
        :param text_queue: 输入文本队列（流式文本块）
        :param audio_queue: 输出音频块队列（bytes）
        :param interrupt_queue: 中断信号队列
        :param voice: TTS 音色名称
        :param sample_rate: 目标采样率（应与播放器一致，通常 24000）
        """
        self.text_queue = text_queue
        self.audio_queue = audio_queue
        self.interrupt_queue = interrupt_queue
        self.voice = voice
        self.sample_rate = sample_rate

        # 文本缓冲区，用于累积不完整的句子
        self.buffer = ""

        # 句子结束标点正则（可根据需要调整）
        self.sentence_delimiters = re.compile(r'[。！？；：!?;:~]')
        self.con = 0
        self.com = 0
        self.queueCount = 1

    async def run(self):
        logger.info("TTS 模块启动")
        try:
            while True:
                # 同时等待文本块和中断信号
                text_chunk = await self._get_with_interrupt()
                self.queueCount +=1
                if text_chunk is None:
                    # 收到中断：清空缓冲区并跳过本次处理
                    logger.debug("TTS 收到中断，清空缓冲区")
                    self.buffer = ""
                    continue

                # 将新文本追加到缓冲区
                self.buffer += text_chunk

                # 切分出完整句子
                await self._process_buffer()

        except asyncio.CancelledError:
            logger.info("TTS 模块被取消")

    async def _get_with_interrupt(self):
        """同时等待文本块和中断信号，返回文本块或 None（中断）"""
        get_task = asyncio.create_task(self.text_queue.get())
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

    async def _process_buffer(self):
        """处理缓冲区，切分句子并合成"""
        while True:
            # 查找缓冲区中第一个句子结束标点的位置
            match = self.sentence_delimiters.search(self.buffer)
            if not match:
                # 没有完整句子，等待更多文本
                logger.debug("未找到句子结束标点，等待更多文本")
                break

            # 找到第一个标点位置（包括标点）
            end_pos = match.end()
            sentence = self.buffer[:end_pos].strip()
            logger.info(f"切分出句子: {repr(sentence)}, 结束位置: {end_pos}")

            # 更新缓冲区，保留剩余部分
            self.buffer = self.buffer[end_pos:].lstrip()

            if sentence:
                # 合成该句子
                await self._synthesize_sentence(sentence)
            else:
                logger.debug("切分出空句子，跳过")

    async def _synthesize_sentence(self, sentence: str):
        """合成单个句子（MP3），解码为 PCM 后放入 audio_queue"""
        if not sentence or not sentence.strip():
            logger.info("跳过空句子")
            return
        logger.info(f"合成句子: {sentence[:30]}...")
        try:
            communicate = edge_tts.Communicate(sentence, self.voice)
        except Exception as e:
            logger.error(f"创建 Communicate 失败: {e}")
            return

        # 边合成边发送音频块
        # 收集该句子的所有 MP3 音频块
        audio_data = b''
        chunk_count = 0
        try:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
                    chunk_count += 1
        except Exception as e:
            logger.error(f"流式合成异常: {e}")
            return
            # 注意：如果合成过程中收到中断，我们需要能够停止这个句子的合成
            # 但 edge_tts 的 stream 不支持中断，我们只能等待它完成。
            # 且我们不再继续合成后续句子。这里无法中断当前句子的合成，因为 edge_tts 不支持。
        logger.info(f"句子合成完成: 总音频大小 {len(audio_data)} 字节, 块数 {chunk_count}")
        if not audio_data:
            logger.error(f"警告: 句子 '{sentence[:50]}...' 未返回任何音频数据!")
            return

        # 将 MP3 解码为 PCM (int16, 单声道, 目标采样率)
        try:
            # 使用 pydub 解码
            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
            audio_segment = audio_segment.set_channels(1).set_frame_rate(self.sample_rate)
            pcm_data = audio_segment.raw_data  # bytes, int16, 小端

            # 将 PCM 数据放入播放队列（可一次性放入，或分块）
            await self.audio_queue.put(pcm_data)
        except Exception as e:
            logger.error(f"解码 MP3 失败: {e}")