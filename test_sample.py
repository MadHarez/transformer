#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试使用高质量数据训练的翻译模型
"""

import os
import torch
from data_utils import sentences_to_ids, build_dict, PAD, DEVICE, BOS, EOS
from model import Transformer

def load_model(model_path):
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        print(f"警告: 模型文件 {model_path} 不存在")
        return None, None, None
    
    print(f"加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        print("模型文件格式不正确")
        return None, None, None
    
    # 从checkpoint获取词汇表大小
    en_total_words = checkpoint.get('en_total_words', 0)
    cn_total_words = checkpoint.get('cn_total_words', 0)
    
    if en_total_words == 0 or cn_total_words == 0:
        print("无法从checkpoint获取词汇表大小")
        return None, None, None
    
    # 创建模型
    model = Transformer(en_total_words, cn_total_words).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch_info = checkpoint.get('epoch', 'Unknown')
    best_loss = checkpoint.get('best_loss', float('inf'))
    print(f"模型epoch: {epoch_info}")
    if best_loss != float('inf'):
        print(f"最佳验证损失: {best_loss:.4f}")
    
    # 返回模型和字典
    return model, checkpoint.get('en_dict'), checkpoint.get('cn_dict')

def translate_sentence(model, sentence, en_dict, cn_index_dict, max_length=60):
    """翻译单个句子"""
    # 将英文句子转换为ID
    words = sentence.lower().split()
    src_ids = [en_dict.get(word, en_dict.get('<unk>', 0)) for word in words]
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(DEVICE)

    # 创建源语言mask
    src_mask = (src_tensor != PAD).unsqueeze(-2)

    # 初始化目标语言（以BOS开始）
    tgt_ids = [BOS]

    with torch.no_grad():
        # 编码源语言
        memory = model.encoder(model.src_embed(src_tensor), src_mask)

        for i in range(max_length):
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(DEVICE)
            tgt_mask = (tgt_tensor != PAD).unsqueeze(-2)
            tgt_mask = torch.tril(torch.ones((1, len(tgt_ids), len(tgt_ids)), device=DEVICE)).bool()

            # 解码
            out = model.decoder(model.tgt_embed(tgt_tensor), memory, src_mask, tgt_mask)
            out = model.generator(out)

            # 获取最后一个词的预测
            next_word_logits = out[0, -1, :]
            next_word_id = torch.argmax(next_word_logits).item()

            # 如果遇到EOS则停止
            if next_word_id == EOS:
                break

            tgt_ids.append(next_word_id)

    # 将ID转换回中文词语
    translated_words = []
    for idx in tgt_ids[1:]:  # 跳过BOS
        if idx in cn_index_dict:
            translated_words.append(cn_index_dict[idx])

    return ''.join(translated_words)

def main():
    print("测试高质量数据训练的翻译模型...")
    
    # 加载模型
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_root, 'best_sample_model.pt')
    
    model, en_dict, cn_dict = load_model(model_path)
    
    if model is None:
        print("模型加载失败")
        return
    
    # 创建反向字典
    cn_index_dict = {v: k for k, v in cn_dict.items()}
    
    print("模型加载成功！开始翻译测试...")
    print("-" * 50)

    # 测试一些训练数据中的句子
    test_sentences = [
        "Hello, how are you today?",
        "I love learning new languages.",
        "The weather is very nice today.",
        "Can you help me with this problem?",
        "This book is very interesting.",
        "We should protect the environment.",
        "Technology has changed our lives.",
        "Education is very important for children.",
        "I want to travel around the world.",
        "Music makes people feel happy."
    ]

    for i, sentence in enumerate(test_sentences, 1):
        translation = translate_sentence(model, sentence, en_dict, cn_index_dict)
        print(f"{i}. 英文: {sentence}")
        print(f"   中文: {translation}")
        print()

    # 交互式翻译
    print("-" * 50)
    print("进入交互式翻译模式（输入 'quit' 退出）")
    while True:
        sentence = input("\\n请输入英文句子: ").strip()
        if sentence.lower() == 'quit':
            break
        if sentence:
            translation = translate_sentence(model, sentence, en_dict, cn_index_dict)
            print(f"翻译结果: {translation}")

if __name__ == "__main__":
    main()
