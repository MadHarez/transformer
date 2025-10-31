#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
创建高质量的示例翻译数据
"""

def create_sample_data():
    """创建一些高质量的英中翻译样本"""
    
    sample_pairs = [
        ("Hello, how are you today?", "你好，你今天怎么样？"),
        ("I love learning new languages.", "我喜欢学习新语言。"),
        ("The weather is very nice today.", "今天天气很好。"),
        ("Can you help me with this problem?", "你能帮我解决这个问题吗？"),
        ("This book is very interesting.", "这本书很有趣。"),
        ("We should protect the environment.", "我们应该保护环境。"),
        ("Technology has changed our lives.", "技术改变了我们的生活。"),
        ("Education is very important for children.", "教育对孩子非常重要。"),
        ("I want to travel around the world.", "我想环游世界。"),
        ("Music makes people feel happy.", "音乐让人们感到快乐。"),
        ("Exercise is good for your health.", "锻炼对你的健康有好处。"),
        ("Reading books can expand your knowledge.", "读书可以扩展你的知识。"),
        ("Friends are very important in life.", "朋友在生活中非常重要。"),
        ("The internet connects people together.", "互联网把人们连接在一起。"),
        ("Cooking is a useful life skill.", "烹饪是一项有用的生活技能。"),
        ("Learning from mistakes helps us grow.", "从错误中学习帮助我们成长。"),
        ("Nature provides many beautiful scenes.", "大自然提供了许多美丽的景色。"),
        ("Hard work usually leads to success.", "努力工作通常会带来成功。"),
        ("Family gives us love and support.", "家庭给我们爱和支持。"),
        ("Time flies when you are having fun.", "当你玩得开心时，时间过得很快。")
    ]
    
    return sample_pairs

def save_sample_data():
    """保存示例数据到文件"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    en_path = os.path.join(project_root, 'data', 'sample.en')
    zh_path = os.path.join(project_root, 'data', 'sample.zh')
    
    pairs = create_sample_data()
    
    with open(en_path, 'w', encoding='utf-8') as f:
        for en, _ in pairs:
            f.write(en + '\n')
    
    with open(zh_path, 'w', encoding='utf-8') as f:
        for _, zh in pairs:
            f.write(zh + '\n')
    
    print(f"已创建示例数据文件：")
    print(f"英文: {en_path}")
    print(f"中文: {zh_path}")
    print(f"包含 {len(pairs)} 对高质量翻译样本")

if __name__ == "__main__":
    import os
    save_sample_data()
