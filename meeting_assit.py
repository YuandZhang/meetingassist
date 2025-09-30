import lazyllm
import os
import sys
from dashscope import MultiModalConversation
import json
from datetime import datetime

# 语音转文本函数
def speech_to_text(audio_file_path):
    """
    使用dashscope多模态对话模型将音频文件转换为文本
    """
    try:
        print(f"开始语音识别，音频文件路径: {audio_file_path}")
        
        # 检查音频文件是否存在
        if not os.path.exists(audio_file_path):
            return f"错误：音频文件不存在，路径为 {audio_file_path}"
        
        print(f"音频文件存在，文件大小: {os.path.getsize(audio_file_path)} 字节")
        
        # 检查是否是目录
        if os.path.isdir(audio_file_path):
            return f"错误：提供的路径是一个目录，不是音频文件: {audio_file_path}"
        
        # 检查文件扩展名
        valid_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        _, ext = os.path.splitext(audio_file_path.lower())
        if ext not in valid_extensions:
            return f"错误：不支持的音频文件格式。支持的格式: {valid_extensions}"
        
        # 构造以file://为前缀的文件路径
        file_path = f"file://{audio_file_path}"
        print(f"构造的文件路径: {file_path}")
        
        # 构造消息内容
        messages = [
            {
                "role": "system", 
                "content": [{"text": "You are a helpful assistant."}]},
            {
                "role": "user",
                # 在 audio 参数中传入以 file:// 为前缀的文件路径
                "content": [{"audio": file_path}, {"text": "音频里在说什么?"}],
            }
        ]
        
        print("准备调用dashscope多模态对话模型...")
        # 调用dashscope多模态对话模型
        response = MultiModalConversation.call(model="qwen-audio-turbo-latest", messages=messages)
        
        print(f"模型响应: {json.dumps(response, ensure_ascii=False, indent=2)}")
        
        # 检查响应是否成功
        if response.get("status_code") != 200:
            return f"语音识别失败，错误码: {response.get('status_code')}, 错误信息: {response.get('message')}"
        
        # 提取识别结果
        if "output" in response and "choices" in response["output"] and len(response["output"]["choices"]) > 0:
            text_result = response["output"]["choices"][0]["message"]["content"][0]["text"]
            print(f"语音识别成功，结果: {text_result}")
            return text_result
        else:
            return "语音识别失败，模型返回格式不正确"
            
    except Exception as e:
        error_msg = f"语音识别过程中出现错误: {str(e)}"
        print(error_msg)  # 打印错误日志
        import traceback
        traceback.print_exc()  # 打印详细的错误堆栈
        # 返回更友好的错误信息
        return "语音识别失败，请确保已正确设置dashscope API密钥，并检查音频格式是否支持。"

# 生成会议纪要函数
def generate_meeting_summary(text_content):
    """
    使用大语言模型生成会议纪要
    """
    try:
        print("开始生成会议纪要...")
        # 创建大语言模型，使用SenseNova支持的模型
        llm = lazyllm.OnlineChatModule(source="sensenova", model='Qwen3-235B')
        
        # 设置提示词，指导模型生成会议纪要
        prompt = '''你是一个专业的会议记录员，你的任务是根据提供的会议内容生成一份结构化的会议纪要。
请按照以下格式输出：
1. 会议主题
2. 会议时间
3. 参会人员
4. 会议要点（按要点分条列出）
5. 决议事项（如果有）
6. 待办事项（如果有，包括负责人和截止日期）

会议内容如下：
{context_str}'''
        
        llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))
        
        # 生成会议纪要
        res = llm({"context_str": text_content})
        print("会议纪要生成完成")
        
        return res
    except Exception as e:
        error_msg = f"生成会议纪要时出现错误: {str(e)}"
        print(error_msg)  # 打印错误日志
        import traceback
        traceback.print_exc()  # 打印详细的错误堆栈
        return "会议纪要生成失败，请稍后重试或检查网络连接。"

# 保存会议纪要为Markdown格式
def save_summary_to_markdown(audio_file_path, text_content, summary):
    """
    将会议纪要保存为Markdown格式文件
    """
    try:
        # 生成文件名，基于音频文件名和当前时间戳
        base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        markdown_filename = f"{base_name}_会议纪要_{timestamp}.md"
        
        # 确保在音频文件相同目录下创建Markdown文件
        output_dir = os.path.dirname(os.path.abspath(audio_file_path))
        if not output_dir:
            output_dir = "."
        markdown_filepath = os.path.join(output_dir, markdown_filename)
        
        # 创建Markdown内容
        markdown_content = f"""# 会议纪要

## 基本信息
- 音频文件: {os.path.basename(audio_file_path)}
- 处理时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 语音识别结果
{text_content}

## 会议纪要
{summary}
"""
        
        # 保存到文件
        with open(markdown_filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"会议纪要已保存为Markdown格式: {markdown_filepath}")
        return markdown_filepath
    except Exception as e:
        error_msg = f"保存会议纪要为Markdown时出现错误: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None

# 处理音频文件并生成会议纪要
def process_meeting_audio(audio_file_path):
    """
    处理音频文件，生成会议纪要
    """
    print({"audio_file_path": audio_file_path})
    try:
        print(f"开始处理音频文件: {audio_file_path}")
        
        # 检查文件路径
        if not audio_file_path:
            return "错误：未提供音频文件路径。"
            
        # 步骤1: 语音转文本
        text_content = speech_to_text(audio_file_path)
        
        # 如果语音识别失败，直接返回错误信息
        if "语音识别失败" in text_content or "错误：" in text_content:
            return text_content
            
        # 步骤2: 生成会议纪要
        summary = generate_meeting_summary(text_content)
        
        # 步骤3: 保存为Markdown格式
        markdown_file = save_summary_to_markdown(audio_file_path, text_content, summary)
        
        # 返回结果
        result = f"语音识别结果:\n{text_content}\n\n会议纪要:\n{summary}"
        if markdown_file:
            result += f"\n\n会议纪要已保存为Markdown文件: {markdown_file}"
        return result
    except Exception as e:
        error_msg = f"处理音频文件时出现错误: {str(e)}"
        print(error_msg)  # 打印错误日志
        import traceback
        traceback.print_exc()  # 打印详细的错误堆栈
        return "处理音频文件时出现未知错误，请稍后重试。"

# 直接运行处理函数
def run_meeting_assistant(audio_file_path):
    """
    运行会议助理功能，处理指定的音频文件
    """
    print("正在启动跨语言会议助理...")
    try:
        # 处理音频文件并生成会议纪要
        result = process_meeting_audio(audio_file_path)
        print("处理完成，结果如下：")
        print("="*50)
        print(result)
        print("="*50)
        return result
    except Exception as e:
        error_msg = f"运行会议助理时出现错误: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()  # 打印详细的错误堆栈
        return error_msg

# 主函数
if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法: python meeting_assit.py <音频文件路径>")
        sys.exit(1)
    
    # 获取音频文件路径
    audio_file_path = sys.argv[1]
    
    # 将相对路径转换为绝对路径以确保正确处理
    audio_file_path = os.path.abspath(audio_file_path)
    
    # 检查文件是否存在
    if not os.path.exists(audio_file_path):
        print(f"错误：音频文件不存在，路径为 {audio_file_path}")
        sys.exit(1)
    
    # 运行会议助理
    run_meeting_assistant(audio_file_path)
    
    