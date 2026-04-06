"""
modules/dialogue_manager.py - 对话管理模块
负责：
- 维护对话历史
- 检索长期记忆
- 构造提示词调用 LLM
- 处理工具调用
- 将新信息存入记忆
"""
import asyncio
import json
import logging
from modules.tools import system_control, screen_reader  # 按需导入工具
from config import MEMORY_ENABLED, MEMORY_PERSIST_DIR, MEMORY_COLLECTION_NAME, \
                   EMBEDDING_MODEL, MEMORY_RETRIEVAL_K, MEMORY_DEVICE
import re
from modules.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class DialogueManager:
    def __init__(self, text_queue, llm_queue, tts_queue, interrupt_queue):
        self.text_queue = text_queue
        self.llm_queue = llm_queue
        self.tts_queue = tts_queue
        self.interrupt_queue = interrupt_queue
        self.history = []  # 对话历史
        self.tools = {}    # 可用工具函数字典
        self.tool_defs = [] # 工具描述（用于LLM函数调用）

        # 初始化记忆管理器
        self.memory_enabled = MEMORY_ENABLED
        if self.memory_enabled:
            self.memory = MemoryManager(
                persist_dir=MEMORY_PERSIST_DIR,
                collection_name=MEMORY_COLLECTION_NAME,
                embedding_model=EMBEDDING_MODEL,
                retrieval_k=MEMORY_RETRIEVAL_K,
                device=MEMORY_DEVICE  # 如有GPU可改为 "cuda"
            )
        else:
            self.memory = None

        self._register_tools()

    def _register_tools(self):
        """注册所有可用工具（可根据配置动态加载）"""
        from config import ENABLE_SYSTEM_TOOL, ENABLE_SCREEN_TOOL

        if ENABLE_SYSTEM_TOOL:
            self.tools["open_application"] = system_control.open_application
            self.tools["list_processes"] = system_control.list_processes
            self.tool_defs.append({
                "type": "function",
                "function": {
                    "name": "open_application",
                    "description": "打开指定的应用程序",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "应用名称"}
                        },
                        "required": ["name"]
                    }
                }
            })
            # 可添加更多工具定义...

        if ENABLE_SCREEN_TOOL:
            self.tools["get_screen_text"] = screen_reader.get_screen_text
            self.tool_defs.append({
                "type": "function",
                "function": {
                    "name": "get_screen_text",
                    "description": "获取当前屏幕上的所有文字（OCR）",
                    "parameters": {"type": "object", "properties": {}}
                }
            })

    async def run(self):
        """主循环：接收用户文本，调用LLM，处理工具调用"""
        logger.info("对话管理 模块启动")
        try:
            while True:
                user_text = await self._get_with_interrupt()
                if user_text is None:
                    logger.info("对话管理收到中断，跳过本轮")
                    # 清空临时状态（如果有）
                    continue
                logger.info(f"用户: {user_text}")

                # 1. 将用户消息加入历史
                self.history.append({"role": "user", "content": user_text})

                # 2. 检索相关记忆
                memory_context = ""
                if self.memory_enabled and self.memory:
                    memories = self.memory.retrieve_memories(user_text)
                    if memories:
                        memory_context = "以下是你知道的关于用户的信息：\n" + "\n".join(
                            f"- {mem}" for mem in memories) + "\n"

                # 3. 构造系统提示和消息列表（系统消息包含记忆上下文）
                role_msg = ("你是我女朋友，对我充满爱意，模拟恋人之间的对话，"
                            "说话简洁，幽默有趣。外向，善于开展话题。"
                            "口语化不要使用表情。")
                messages = [{"role": "system", "content": role_msg},
                            *self.history[-12:]# 最近5轮对话
                            ]
                if memory_context:
                    messages.append({"role": "system", "content": memory_context})

                # 4. 调用LLM（支持工具调用）
                response = await self._call_llm(messages, self.tool_defs)

                # 5. 处理LLM响应（可能是工具调用或文本回复）
                if response.get("type") == "tool_call":
                    # 处理工具调用
                    tool_name = response["tool_call"]["name"]
                    args = response["tool_call"]["arguments"]
                    tool_call_id = response["tool_call"]["id"]
                    logger.info(f"调用工具: {tool_name}, 参数: {args}")

                    try:
                        result = self.tools[tool_name](**args)
                        result_str = json.dumps(result, ensure_ascii=False)
                    except Exception as e:
                        result_str = f"工具调用失败: {e}"

                    # 将工具结果作为 tool 消息加入历史
                    self.history.append({
                        "role": "tool",
                        "content": result_str,
                        "tool_call_id": tool_call_id
                    })

                    # 再次调用LLM生成最终回复（无工具调用）
                    final_response = await self._call_llm(self.history[-10:])
                    reply = final_response.get("content", "")
                else:
                    # 普通文本回复
                    reply = response.get("content", "")

                # 6. 如果有回复，加入历史并送入TTS队列
                if reply:
                    logger.info(f"AI: {reply}")
                    self.history.append({"role": "assistant", "content": reply})
                    # await self.tts_queue.put(reply)

                # 7. 提取用户文本中的事实并存入记忆
                if self.memory_enabled and self.memory:
                    self.memory.add_memory_unique(user_text, threshold=0.95)  # 阈值可配置
        except asyncio.CancelledError:
            logger.info("对话管理模块被取消")

    async def _get_with_interrupt(self):
        """同时等待用户文本和中断信号"""
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
            # 收到中断，消耗信号并返回 None
            interrupt_task.result()
            return None
        else:
            return get_task.result()


    async def _call_llm(self, messages, tools=None):
        """将请求放入LLM队列并等待结果（同步方式）"""
        # 创建一个一次性队列来接收LLM的回复
        resp_queue = asyncio.Queue()
        await self.llm_queue.put({
            "messages": messages,
            "tools": tools,
            "response_queue": resp_queue
        })
        return await resp_queue.get()