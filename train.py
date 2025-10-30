# train.py

import os
import torch
import torch.optim as optim
from torch import nn

from data_utils import sentences_to_ids, split_batch, load_data, build_dict, PAD, DEVICE
from model import Transformer

# 获取项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))

EPOCHS = 20
LEARNING_RATE = 0.0001


def train_epoch(model, train_batches, optimizer, criterion):
    """
    单 epoch 训练。
    :param model: Transformer 模型
    :param train_batches: 训练批次列表
    :param optimizer: 优化器
    :param criterion: 损失函数
    :return: 平均损失
    """
    model.train()
    total_loss = 0
    for batch in train_batches:
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = criterion(out.contiguous().view(-1, out.size(-1)), batch.trg_y.contiguous().view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(train_batches)


def main_train(en_total_words, cn_total_words, train_batches):
    """
    主训练函数。
    """
    model = Transformer(en_total_words, cn_total_words).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_batches, optimizer, criterion)
        print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {loss:.4f}')
        torch.save(model.state_dict(), 'model.pt')  # 保存模型


# 示例调用（在脚本底部）
if __name__ == "__main__":
    # 根据你的实际文件名修正路径
    en_path = os.path.join(project_root, 'data', 'MDN_Web_Docs.en-zh_CN.en')
    zh_path = os.path.join(project_root, 'data', 'MDN_Web_Docs.en-zh_CN.zh_CN')

    # 调试：检查文件是否存在
    print(f"英文文件路径: {en_path}")
    print(f"中文文件路径: {zh_path}")
    print(f"英文文件存在: {os.path.exists(en_path)}")
    print(f"中文文件存在: {os.path.exists(zh_path)}")

    if not os.path.exists(en_path) or not os.path.exists(zh_path):
        print("错误：数据文件不存在！")
        print("请检查文件名是否正确")
        exit(1)

    en_sentences, cn_sentences = load_data(en_path, zh_path)
    en_dict, en_total_words, en_index_dict = build_dict(en_sentences)
    cn_dict, cn_total_words, cn_index_dict = build_dict(cn_sentences)
    en_ids, cn_ids = sentences_to_ids(en_sentences, cn_sentences, en_dict, cn_dict)
    train_batches = split_batch(en_ids, cn_ids, batch_size=16)  # 减小 batch_size 以适应大型数据集
    main_train(en_total_words, cn_total_words, train_batches)