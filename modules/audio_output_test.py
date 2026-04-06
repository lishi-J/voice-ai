import asyncio
import edge_tts


async def test():
    text = "你好，这是一个测试。"
    voice = "zh-CN-XiaoyiNeural"
    communicate = edge_tts.Communicate(text, voice)

    audio_data = b''
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
            print(f"收到块，大小 {len(chunk['data'])}")

    if audio_data:
        print(f"成功收到音频，总大小 {len(audio_data)} 字节")
        with open("test_output.mp3", "wb") as f:
            f.write(audio_data)
        print("已保存为 test_output.mp3")
    else:
        print("未收到任何音频数据！")


if __name__ == "__main__":
    asyncio.run(test())