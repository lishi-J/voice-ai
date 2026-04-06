# voice-ai
**项目简介**

开发了一个模块化的实时语音对话AI系统，支持自然语音交互、长期记忆和任务执行（如操作电脑、控制智能家居）。系统采用事件驱动架构，实现低延迟和用户打断功能。

**技术栈**

- **异步框架**：Python asyncio + 队列，实现模块间解耦与高并发处理
- **语音识别**：本地部署 Qwen3-ASR-0.6B（Hugging Face Transformers），结合 Silero VAD 实时切分语音句子
- **大语言模型**：DeepSeek API（兼容OpenAI接口），支持函数调用，实现对话理解与工具调度
- **语音合成**：Edge TTS 在线服务（微软免费接口），流式输出 MP3 并实时解码为 PCM，支持多语言音色
- **长期记忆**：基于 RAG 的向量数据库（Chroma + sentence-transformers），存储用户信息并实现相似度检索
- **音频处理**：sounddevice 实现低延迟麦克风采集与扬声器播放，支持中断信号广播

**系统设计**

- **架构设计**：设计异步管道式架构（AudioInput → ASR → DialogueManager → LLM → TTS → AudioOutput），各模块通过队列通信，独立可扩展；增加 interrupt_queue 实现全局打断，用户说话时 AI 立即停止播放。
- **语音识别优化**：集成 Silero VAD 进行语音活动检测，动态切分完整句子送入 ASR，避免静音段无效识别；块大小自适应，识别准确率提升。
- **对话与记忆**：实现基于 LangChain 的 RAG 记忆模块，用户个人信息自动存入向量库，对话时检索相关记忆注入上下文；支持工具调用（如打开应用、获取屏幕文字），扩展系统控制能力。
- **实时语音合成**：采用 Edge TTS 在线接口，流式接收 MP3 音频，使用 pydub 解码为 24kHz PCM 后实时播放；通过按标点切分句子实现自然停顿，并支持中断时清空缓冲区。
- **日志与调试**：配置多输出日志（终端+文件），按日期轮转，便于线上问题追踪。

voice_ai/
├── main.py                 # 程序入口：初始化模块、启动事件循环
├── config.py               # 统一配置（模型路径、API密钥、队列大小等）
├── requirements.txt        # 依赖库
├── .env.example            # 环境变量示例（可选）
├── modules/                # 核心功能模块
│   ├── __init__.py
│   ├── audio_input.py      # 麦克风采集 + VAD + 中断检测
│   ├── asr.py              # 语音识别（流式）
│   ├── dialogue_manager.py # 对话管理 + 工具调用调度
│   ├── llm.py              # 大语言模型调用（支持流式）
│   ├── tts.py              # 语音合成（流式）
│   ├── audio_output.py     # 音频播放 + 中断响应
│   └── tools/              # 扩展工具集（按需添加）
│       ├── __init__.py
│       ├── system_control.py  # 系统命令、进程管理等
│       └── screen_reader.py   # 屏幕截图、OCR
├── utils/                  # 辅助工具
│   ├── audio_utils.py      # 音频格式转换、重采样
│   └── queue_utils.py      # 队列封装（监控、中断信号）
└── logs/                   # 日志目录（自动生成）
