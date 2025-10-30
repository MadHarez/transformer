
import os
import torch
from torch import nn
from data_utils import sentences_to_ids, split_batch, load_data, build_dict, PAD, DEVICE
from model import Transformer


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
            total_loss += loss.item() * batch.ntokens
            total_tokens += batch.ntokens

    return total_loss / total_tokens


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))

    # 加载数据
    en_path = os.path.join(project_root, 'data', 'MDN_Web_Docs.en-zh_CN.en')
    zh_path = os.path.join(project_root, 'data', 'MDN_Web_Docs.en-zh_CN.zh_CN')

    en_sentences, cn_sentences = load_data(en_path, zh_path)
    en_dict, en_total_words, en_index_dict = build_dict(en_sentences)
    cn_dict, cn_total_words, cn_index_dict = build_dict(cn_sentences)

    # 分割训练集和测试集
    split_idx = int(0.8 * len(en_sentences))  # 80% 训练，20% 测试
    train_en = en_sentences[:split_idx]
    train_cn = cn_sentences[:split_idx]
    test_en = en_sentences[split_idx:]
    test_cn = cn_sentences[split_idx:]

    # 转换为ID
    train_en_ids, train_cn_ids = sentences_to_ids(train_en, train_cn, en_dict, cn_dict)
    test_en_ids, test_cn_ids = sentences_to_ids(test_en, test_cn, en_dict, cn_dict)

    # 创建批次
    test_batches = split_batch(test_en_ids, test_cn_ids, batch_size=16)

    # 加载模型
    model_path = os.path.join(project_root, 'model.pt')
    model = Transformer(en_total_words, cn_total_words).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    # 评估
    test_loss = evaluate_model(model, test_batches, criterion)
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集困惑度: {torch.exp(torch.tensor(test_loss)):.4f}")


if __name__ == "__main__":
    main()