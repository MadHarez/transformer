# train.py

import os
import torch
import torch.optim as optim
from torch import nn

from data_utils import sentences_to_ids, split_batch, load_data, build_dict, PAD, DEVICE
from model import Transformer

# 获取项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))

EPOCHS = 20  # 继续训练的轮数
LEARNING_RATE = 0.0001
RESUME_TRAINING = True  # 设置为 True 来继续训练，False 则从头开始
MODEL_PATH = os.path.join(project_root, 'model.pt')


def train_epoch(model, train_batches, optimizer, criterion):
    """
    单 epoch 训练。
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


def load_existing_model(model_path, en_total_words, cn_total_words):
    """加载已训练的模型"""
    model = Transformer(en_total_words, cn_total_words).to(DEVICE)
    if os.path.exists(model_path):
        print(f"加载已训练模型: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print("未找到已训练模型，从头开始训练")
    return model


def main_train(en_total_words, cn_total_words, train_batches, start_epoch=0):
    """
    主训练函数。
    """
    # 加载模型
    if RESUME_TRAINING and os.path.exists(MODEL_PATH):
        model = load_existing_model(MODEL_PATH, en_total_words, cn_total_words)
    else:
        model = Transformer(en_total_words, cn_total_words).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    # 如果继续训练，可以加载优化器状态（可选）
    # optimizer_path = os.path.join(project_root, 'optimizer.pt')
    # if RESUME_TRAINING and os.path.exists(optimizer_path):
    #     optimizer.load_state_dict(torch.load(optimizer_path))

    for epoch in range(start_epoch, start_epoch + EPOCHS):
        loss = train_epoch(model, train_batches, optimizer, criterion)
        print(f'Epoch {epoch + 1}/{start_epoch + EPOCHS}, Loss: {loss:.4f}')

        # 保存模型和优化器状态
        torch.save(model.state_dict(), MODEL_PATH)
        # torch.save(optimizer.state_dict(), 'optimizer.pt')  # 可选保存优化器状态


def main():
    # 根据你的实际文件名修正路径
    en_path = os.path.join(project_root, 'data', 'MDN_Web_Docs.en-zh_CN.en')
    zh_path = os.path.join(project_root, 'data', 'MDN_Web_Docs.en-zh_CN.zh_CN')

    print(f"英文文件路径: {en_path}")
    print(f"中文文件路径: {zh_path}")
    print(f"英文文件存在: {os.path.exists(en_path)}")
    print(f"中文文件存在: {os.path.exists(zh_path)}")

    if not os.path.exists(en_path) or not os.path.exists(zh_path):
        print("错误：数据文件不存在！")
        exit(1)

    en_sentences, cn_sentences = load_data(en_path, zh_path)
    en_dict, en_total_words, en_index_dict = build_dict(en_sentences)
    cn_dict, cn_total_words, cn_index_dict = build_dict(cn_sentences)
    en_ids, cn_ids = sentences_to_ids(en_sentences, cn_sentences, en_dict, cn_dict)
    train_batches = split_batch(en_ids, cn_ids, batch_size=16)

    # 计算起始epoch（用于显示）
    start_epoch = 0
    if RESUME_TRAINING and os.path.exists(MODEL_PATH):
        # 这里可以记录之前的训练轮数，或者从文件名中解析
        # 简单起见，我们假设从上次停止的地方继续
        start_epoch = 20  # 如果你之前训练了20轮，这里设为20

    main_train(en_total_words, cn_total_words, train_batches, start_epoch)


if __name__ == "__main__":
    main()