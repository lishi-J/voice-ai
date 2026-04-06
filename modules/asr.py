import asyncio
import logging

import torch
from qwen_asr import Qwen3ASRModel

import config
from utils.audio_utils import int16_to_float32

logger = logging.getLogger(__name__)

class ASR:
    def __init__(self, audio_queue, text_queue, interrupt_queue, sample_rate=config.SAMPLE_RATE):
        """
                :param audio_queue: 输入音频块队列（bytes，int16格式，16kHz）
                :param text_queue: 输出识别文本队列
                :param interrupt_queue: 中断信号队列
                :param sample_rate: 音频采样率，必须与模型要求一致（16kHz）
                """
        self.audio_queue = audio_queue
        self.text_queue = text_queue
        self.interrupt_queue = interrupt_queue
        self.sample_rate = sample_rate

        logger.info("加载ASR模型：Qwen/Qwen3-ASR-0.6B")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-0.6B",
            dtype=torch.bfloat16,
            device_map="cuda:0",
            # attn_implementation="flash_attention_2",
            max_inference_batch_size=32,
            # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
            max_new_tokens=256,  # Maximum number of tokens to generate. Set a larger value for long audio input.
            forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
            forced_aligner_kwargs=dict(
                dtype=torch.bfloat16,
                device_map="cuda:0",
                # attn_implementation="flash_attention_2",
            )
        )
        logger.info("ASR模型加载完成")

    async def run(self):
        """使用本地的Qwen3ASR模型进行语音识别"""
        logger.info("ASR 模块启动")
        try:
            while True:
                # logger.info("在循环中")
                # 等待音频块或中断信号
                audio_bytes = await self._get_with_interrupt()
                if audio_bytes is None:
                    # 收到中断，取消当前识别（如果有）
                    logger.debug("ASR 收到中断，跳过当前识别")
                    continue

                # 执行识别
                await self._recognize(audio_bytes)
        except asyncio.CancelledError:
            logger.info("ASR 模块被取消")

    async def _recognize(self, audio_bytes):
        logger.info("开始识别!")
        try:
            audio_np = int16_to_float32(audio_bytes)
            audio_input = (audio_np, self.sample_rate)  # 构造元组
            results = self.model.transcribe(
                audio=audio_input,
                language=None, # can also be set to None for automatic language detection
                return_time_stamps=False,
            )
            if isinstance(results, list) and len(results) == 1:
                logger.info("识别成功！")
                logger.info(results[0].text)
                if len(results[0].text ) > 0:
                    await self.text_queue.put(results[0].text)
        except Exception as e:
            logger.error(f"ASR 识别失败: {e}")

    async def _get_with_interrupt(self):
        # logger.info("在中断算法中")
        """同时等待音频和中断，返回音频字节或 None（中断）"""
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