import os
import torch
from torch import nn
import math
from data_utils import sentences_to_ids, split_batch, load_data, build_dict, PAD, DEVICE
from model import Transformer

def load_model(model_path, en_total_words, cn_total_words):
    """加载训练好的模型"""
    model = Transformer(en_total_words, cn_total_words).to(DEVICE)
    if os.path.exists(model_path):
        print(f"加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch_info = checkpoint.get('epoch', 'Unknown')
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"模型epoch: {epoch_info}")
            if best_loss != float('inf'):
                print(f"最佳验证损失: {best_loss:.4f}")
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"警告: 模型文件 {model_path} 不存在")
        return None
    model.eval()
    return model

def evaluate_model(model, test_batches, criterion):
    """评估模型在测试集上的表现"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in test_batches:
            out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            loss = criterion(out.contiguous().view(-1, out.size(-1)),
                             batch.trg_y.contiguous().view(-1))
            total_loss += loss.item()
            total_tokens += batch.ntokens

    return total_loss / len(test_batches)

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))

    # 加载数据
    en_path = os.path.join(project_root, 'data', 'MDN_Web_Docs.en-zh_CN.en')
    zh_path = os.path.join(project_root, 'data', 'MDN_Web_Docs.en-zh_CN.zh_CN')

    print("加载数据...")
    en_sentences, cn_sentences = load_data(en_path, zh_path)
    print(f"总数据量: {len(en_sentences)}对句子")
    
    # 构建字典
    en_dict, en_total_words, en_index_dict = build_dict(en_sentences)
    cn_dict, cn_total_words, cn_index_dict = build_dict(cn_sentences)
    print(f"英文词汇表大小: {en_total_words}")
    print(f"中文词汇表大小: {cn_total_words}")
    
    # 数据分割：80%训练，10%验证，10%测试
    total_size = len(en_sentences)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    
    train_en = en_sentences[:train_size]
    train_cn = cn_sentences[:train_size]
    val_en = en_sentences[train_size:train_size + val_size]
    val_cn = cn_sentences[train_size:train_size + val_size]
    test_en = en_sentences[train_size + val_size:]
    test_cn = cn_sentences[train_size + val_size:]
    
    print(f"训练集: {len(train_en)}对")
    print(f"验证集: {len(val_en)}对")
    print(f"测试集: {len(test_en)}对")
    
    # 转换为ID
    test_en_ids, test_cn_ids = sentences_to_ids(test_en, test_cn, en_dict, cn_dict)
    
    # 创建批次
    test_batches = split_batch(test_en_ids, test_cn_ids, batch_size=32, shuffle=False)
    print(f"测试批次数: {len(test_batches)}")

    # 尝试加载最佳模型，如果不存在则加载普通模型
    best_model_path = os.path.join(project_root, 'best_model.pt')
    model_path = os.path.join(project_root, 'model.pt')
    
    model = None
    if os.path.exists(best_model_path):
        model = load_model(best_model_path, en_total_words, cn_total_words)
    elif os.path.exists(model_path):
        model = load_model(model_path, en_total_words, cn_total_words)
    
    if model is None:
        print("错误: 没有找到可用的模型文件")
        return

    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    # 评估
    print("\n开始评估...")
    test_loss = evaluate_model(model, test_batches, criterion)
    perplexity = torch.exp(torch.tensor(test_loss))
    
    print(f"\n=== 评估结果 ===")
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集困惑度: {perplexity:.4f}")
    
    # 添加一些示例翻译
    print("\n=== 示例翻译 ===")
    from inference import translate_sentence
    
    test_sentences = [
        "hello world",
        "how are you",
        "this is a test",
        "good morning",
        "thank you very much"
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        translation = translate_sentence(model, sentence, en_dict, cn_index_dict)
        print(f"{i}. 英文: {sentence}")
        print(f"   中文: {translation}")
        print()

if __name__ == "__main__":
    main()