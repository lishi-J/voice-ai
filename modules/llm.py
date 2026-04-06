"""
modules/llm.py - 大语言模型调用模块
接收来自对话管理模块的请求，调用 OpenAI API 或兼容接口，返回响应。
支持工具调用和普通文本生成，并处理中断信号。
"""
import asyncio
import logging

import json
from openai import OpenAI  # 以OpenAI为例，可替换为其他SDK
from config import LLM_API_KEY, LLM_API_BASE, LLM_MODEL

logger = logging.getLogger(__name__)

class LLM:
    def __init__(self, input_queue, output_queue, interrupt_queue):
        self.input_queue = input_queue
        self.output_queue = output_queue  # 用于直接输出文本（非工具调用）
        self.interrupt_queue = interrupt_queue
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)

    async def run(self):
        logger.info("llm 模块启动")
        try:
            while True:
                request = await self.input_queue.get()
                messages = request.get("messages")
                logger.info(messages)
                tools = request.get("tools")
                resp_queue = request.get("response_queue")

                # 处理中断：若收到中断信号，则取消当前生成
                interrupt_task = asyncio.create_task(self.interrupt_queue.get())
                try:
                    logger.info("LLM交互开始")
                    fulling_content = ""
                    if tools:
                        # 带工具调用的请求（使用非流式，因为需要解析工具调用）
                        response = self.client.chat.completions.create(
                            model=LLM_MODEL,
                            messages=messages,
                            tools=tools,
                            tool_choice="auto"
                        )
                        choice = response.choices[0]
                        if choice.finish_reason == "tool_calls":
                            tool_call = choice.message.tool_calls[0]
                            result = {
                                "type": "tool_call",
                                "tool_call": {
                                    "name": tool_call.function.name,
                                    "arguments": json.loads(tool_call.function.arguments),
                                    "id": tool_call.id
                                }
                            }
                        else:
                            await self.output_queue.put(choice.message.content)
                    else:
                        # 纯文本请求，可以流式输出到TTS
                        response = self.client.chat.completions.create(
                            model=LLM_MODEL,
                            messages=messages,
                            stream=True
                        )

                        for chunk in response:
                            # 检查中断（每次收到块时快速检查）
                            if interrupt_task.done():
                                # 已收到中断信号，停止生成
                                logger.debug("流式生成被中断")
                                break
                            """
                            if chunk.choices[0].delta.reasoning_content:
                                token = chunk.choices[0].delta.reasoning_content
                                reasoning_content += token
                            """
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                await self.output_queue.put(content)
                                fulling_content += content
                        logger.info(fulling_content)
                    # 正常结束，返回完整内容，标记已流式输出
                    await resp_queue.put({"type": "text", "content": fulling_content, "streamed": True})
                except asyncio.CancelledError:
                    logger.debug("LLM 生成任务被取消")
                except Exception as e:
                    logger.error(f"LLM 调用失败: {e}")
                finally:
                    interrupt_task.cancel()
                    try:
                        await interrupt_task
                    except asyncio.CancelledError:
                        pass

        except asyncio.CancelledError:
            logger.info("LLM 模块被取消")
