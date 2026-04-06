import asyncio
import logging
import logging.handlers
import os

from modules.audio_input import AudioInput
from modules.asr import ASR
from modules.dialogue_manager import DialogueManager
from modules.llm import LLM
from modules.tts import TTS
from modules.audio_output import AudioOutput
import config
logging.basicConfig(level=config.LOG_LEVEL,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

async def main():
    logging.info("启动语音助手系统")
    setup_logging()

    # 创建队列（所有模块间通过队列通信）
    audio_in_queue = asyncio.Queue(maxsize=100)   # 原始音频块 -> ASR
    text_in_queue = asyncio.Queue()                # ASR文本 -> 对话管理
    llm_in_queue = asyncio.Queue()                  # 对话历史/工具请求 -> LLM
    tts_in_queue = asyncio.Queue()                  # 待合成文本 -> TTS
    audio_out_queue = asyncio.Queue()                # 合成音频块 -> 播放器

    # 中断信号队列（用于打断播放和生成）
    interrupt_queue = asyncio.Queue()

    # 实例化模块（依赖注入）
    audio_in = AudioInput(
        audio_queue=audio_in_queue,
        interrupt_queue=interrupt_queue,
        rate=config.SAMPLE_RATE,
        chunk=config.CHUNK_SIZE,  # 50ms 帧数
        silence_timeout_ms=config.silence_timeout_ms
    )

    asr = ASR(
        audio_queue=audio_in_queue,
        text_queue=text_in_queue,
        interrupt_queue=interrupt_queue
    )

    dm = DialogueManager(
        text_queue=text_in_queue,
        llm_queue=llm_in_queue,
        tts_queue=tts_in_queue,
        interrupt_queue=interrupt_queue
    )

    llm = LLM(
        input_queue=llm_in_queue,
        output_queue=tts_in_queue,
        interrupt_queue=interrupt_queue
    )

    tts = TTS(
        text_queue=tts_in_queue,
        audio_queue=audio_out_queue,
        interrupt_queue=interrupt_queue
    )
    
    audio_out = AudioOutput(
        audio_queue=audio_out_queue,
        interrupt_queue=interrupt_queue
    )

    # 并发运行所有模块
    await asyncio.gather(
        audio_in.run(),
        asr.run(),
        dm.run(),
        llm.run(),
        tts.run(),
        audio_out.run()
    )

def setup_logging():
    """配置日志：同时输出到终端和文件"""
    # 创建日志目录（如果不存在）
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))  # 从 config 读取级别

    # 清除已有的处理器（避免重复）
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 日志格式
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. 终端处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 2. 文件处理器（按天轮转）
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=os.path.join(log_dir, 'app.log'),
        when='midnight',      # 每天午夜轮转
        interval=1,
        backupCount=7,        # 保留最近7个日志文件
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 可选：降低某些第三方库的日志级别，避免干扰
    logging.getLogger('sounddevice').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    logging.info("日志系统初始化完成")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("程序被用户中断")