import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent

# 音频参数
SAMPLE_RATE = 16000      # 采样率（Hz）
CHUNK_SIZE = 3200        # 每次读取的音频块大小（对应100ms @16kHz）
VAD_THRESHOLD = 0.7      # VAD语音门限
silence_timeout_ms = 2000 # 静音超时，超过此时长且无新语音则结束句子
min_speech_duration_ms = 100 # 最小语音段长度
min_silence_duration_ms=500  # 用于句子切分的静音时长

# 队列大小（防止内存溢出）
MAX_QUEUE_SIZE = 100

# 模型配置（可根据实际使用的引擎修改）
ASR_TYPE = "local"         # "api" 或 "local"
ASR_ENDPOINT = "wss://your-asr-endpoint"
ASR_API_KEY = os.getenv("ASR_API_KEY", "")

LLM_TYPE = "api"         # "api" 或 "local"
LLM_MODEL = "deepseek-chat"
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "sk-d95e99f5908d4b2a8e1e0f3b5981ce23")
LLM_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com")

#TTS_TYPE = "edge"        # "edge"（免费在线）或 "local"
#TTS_VOICE = "zh-CN-XiaoyiNeural"
#TTS_SAMPLE_RATE = 24000 #目标采样率（应与播放器一致，通常 24000）

# TTS 配置 (本地 Qwen)
TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
TTS_VOICE = "female, young, clear"  # 声音描述（VoiceDesign 版本支持自然语言）
TTS_SAMPLE_RATE = 24000  # 必须与 audio_output 一致
TTS_DEVICE = "cuda"  # 或 "cpu"

# 添加记忆相关配置
MEMORY_ENABLED = True
MEMORY_COLLECTION_NAME = "user_memory"
MEMORY_PERSIST_DIR = "./memory_db"
MEMORY_RETRIEVAL_K = 3          # 每次检索返回的最相关记忆条数
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 根据你的语言选择
MEMORY_DEVICE = "cuda" # 如有GPU可改为 "cuda"

# 工具开关（按需启用）
ENABLE_SYSTEM_TOOL = False
ENABLE_SCREEN_TOOL = False

# 日志级别
LOG_LEVEL = "INFO"