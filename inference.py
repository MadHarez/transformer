# inference.py

import os
import torch
from data_utils import sentences_to_ids, build_dict, load_data, PAD, DEVICE, BOS, EOS
from model import Transformer


def load_model(model_path, en_total_words, cn_total_words):
    """加载训练好的模型"""
    model = Transformer(en_total_words, cn_total_words).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


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

    return ' '.join(translated_words)


def main():
    # 加载字典和数据
    project_root = os.path.dirname(os.path.abspath(__file__))
    en_path = os.path.join(project_root, 'data', 'MDN_Web_Docs.en-zh_CN.en')
    zh_path = os.path.join(project_root, 'data', 'MDN_Web_Docs.en-zh_CN.zh_CN')

    en_sentences, cn_sentences = load_data(en_path, zh_path)
    en_dict, en_total_words, en_index_dict = build_dict(en_sentences)
    cn_dict, cn_total_words, cn_index_dict = build_dict(cn_sentences)

    # 加载模型
    model_path = os.path.join(project_root, 'model.pt')
    model = load_model(model_path, en_total_words, cn_total_words)

    print("模型加载成功！开始翻译测试...")
    print("-" * 50)

    # 测试一些句子
    test_sentences = [
        "hello world",
        "how are you",
        "this is a test",
        "good morning",
        "thank you very much",
        "what is your name"
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
        sentence = input("\n请输入英文句子: ").strip()
        if sentence.lower() == 'quit':
            break
        if sentence:
            translation = translate_sentence(model, sentence, en_dict, cn_index_dict)
            print(f"翻译结果: {translation}")


if __name__ == "__main__":
    main()