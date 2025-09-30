from dashscope import MultiModalConversation

# 将 ABSOLUTE_PATH/welcome.mp3 替换为本地音频的绝对路径，
# 本地文件的完整路径必须以 file:// 为前缀，以保证路径的合法性，例如：file:///home/images/test.mp3
audio_file_path = "file:///Users/umeikousmacbook/Library/Mobile Documents/com~apple~CloudDocs/Trae/商汤/agentdemo/跨语言会议助理/长沙区块链产业园.wav"
messages = [
    {
        "role": "system", 
        "content": [{"text": "You are a helpful assistant."}]},
    {
        "role": "user",
        # 在 audio 参数中传入以 file:// 为前缀的文件路径
        "content": [{"audio": audio_file_path}, {"text": "音频里在说什么?"}],
    }
]

response = MultiModalConversation.call(model="qwen-audio-turbo-latest", messages=messages)
print("输出结果为：")
print(response["output"]["choices"][0]["message"].content[0]["text"])