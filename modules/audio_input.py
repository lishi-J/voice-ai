"""
modules/audio_input.py - 麦克风采集 + VAD 语音活动检测
职责：
- 从麦克风采集音频块
- 使用 Silero VAD 检测语音起止
- 当语音开始时，向 interrupt_queue 发送中断信号
- 当语音结束时，将累积的整个句子（音频 bytes）放入 audio_queue
"""
import asyncio
import sounddevice as sd
import logging
import time
import config

# 可选VAD库，这里用占位函数
from silero_vad import load_silero_vad, get_speech_timestamps  # 需要安装 silero-vad

from utils.audio_utils import int16_to_float32

logger = logging.getLogger(__name__)

class AudioInput:
    def __init__(self, audio_queue, interrupt_queue,
                 rate=config.SAMPLE_RATE,
                 chunk=config.CHUNK_SIZE,
                 vad_threshold=config.VAD_THRESHOLD,
                 min_speech_duration_ms=config.min_speech_duration_ms,
                 min_silence_duration_ms=config.min_speech_duration_ms,  # 用于句子切分的静音时长
                 silence_timeout_ms=config.silence_timeout_ms,
                 device=None
                 ):
        """
               :param audio_queue: 输出队列，存放完整的语音句子（bytes, int16）
               :param interrupt_queue: 中断信号队列，检测到语音开始时发送 "interrupt"
               :param rate: 采样率（必须与模型一致，16kHz）
               :param chunk: 每次读取的音频块大小（字节数，对应 100ms @16kHz 是 3200？实际 chunk=1600 对应 100ms 若为16位单声道：1600/2=800采样点，800/16000=0.05s=50ms。可调整）
               :param vad_threshold: VAD 阈值
               :param min_speech_duration_ms: 最小语音段长度
               :param min_silence_duration_ms: 用于 VAD 切分的最小静音长度
               :param silence_timeout_ms: 静音超时，超过此时长且无新语音则结束句子
               :param device: 输入设备ID（None 表示默认设备）
               """
        self.audio_queue = audio_queue
        self.interrupt_queue = interrupt_queue
        self.rate = rate
        self.chunk = chunk
        self.device = device

        # VAD 参数
        self.vad_threshold = vad_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.silence_timeout = silence_timeout_ms / 1000.0
        # 加载 VAD 模型（Silero VAD）
        self.vad_model = load_silero_vad()
        logger.info("VAD 模型加载完成")

        # 状态变量
        self.sentence_buffer = bytearray()  # 当前句子累积音频
        self.last_voice_time = None  # 最后一次检测到语音的时间戳
        self.is_in_speech = False  # 当前是否处于语音段

    async def run(self):
        """主循环：采集音频、VAD 检测、句子切分"""
        stream = sd.InputStream(
            samplerate=self.rate,
            blocksize=self.chunk,
            device=self.device,
            channels=1,
            dtype='int16',
            latency='low'
        )
        stream.start()
        logger.info("麦克风开启")
        loop = asyncio.get_event_loop()
        try:
            while True:
                # 从麦克风读取音频块（阻塞，放入线程池执行）
                # stream.read 返回 (data, overflow)，data 形状为 (chunk, 1)
                data, overflow = await loop.run_in_executor(None, stream.read, self.chunk)
                if overflow:
                    logger.warning("音频缓冲区溢出，可能有数据丢失")
                audio_bytes = data.tobytes()
                # 转换为 numpy float32 用于 VAD
                audio_np = int16_to_float32(audio_bytes)
                audio_np = audio_np.flatten()  # 确保一维

                # 2. VAD 检测当前块是否有语音
                # logger.info("开始VAD检测")
                speech_ts = get_speech_timestamps(
                    audio_np,
                    self.vad_model,
                    sampling_rate=self.rate,
                    threshold=self.vad_threshold,
                    min_speech_duration_ms=self.min_speech_duration_ms,
                    return_seconds=True
                )
                current_has_speech = len(speech_ts) > 0

                # 3. 处理语音开始
                if current_has_speech and not self.is_in_speech:
                    logger.info("语音处理开始")
                    # 新语音段开始
                    self.is_in_speech = True
                    self.last_voice_time = time.time()
                    # 发送中断信号（打断 TTS 播放等）
                    await self.interrupt_queue.put("interrupt")
                    logger.info("发送中断信号")

                # 4. 持续累积音频到句子缓冲区（只要在语音段内或句子未结束）
                if self.is_in_speech or len(self.sentence_buffer) > 0:
                    self.sentence_buffer.extend(audio_bytes)
                    logger.debug("放入缓冲区")

                # 5. 更新最后语音时间
                if current_has_speech:
                    self.last_voice_time = time.time()
                    logger.debug("记录更新最后语音时间")

                # 6. 检测语音结束（当前无语音，且之前有语音）
                if not current_has_speech and self.is_in_speech:
                    self.is_in_speech = False
                    logger.debug("语音结束")

                # 7. 判断是否应该结束当前句子（静音超时）
                if self.last_voice_time is not None:
                    silence_duration = time.time() - self.last_voice_time
                    if silence_duration > self.silence_timeout and len(self.sentence_buffer) > 0:
                        # 句子结束
                        logger.debug(f"静音超时 {silence_duration:.2f}s，句子结束")
                        await self._finish_sentence()
        except asyncio.CancelledError:
            pass
        finally:
            stream.stop()
            stream.close()
            logger.info("麦克风已关闭")

    async def _finish_sentence(self):
        """将当前句子放入 audio_queue 并清空缓冲区"""
        if len(self.sentence_buffer) > 0:
            # 放入队列（bytes 格式，int16）
            await self.audio_queue.put(bytes(self.sentence_buffer))
            logger.info(f"放入句子，长度 {len(self.sentence_buffer)} 字节")
        self.sentence_buffer.clear()
        self.last_voice_time = None
        self.is_in_speech = False