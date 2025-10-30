import os
import torch
import torch.optim as optim
from torch import nn
import json

from data_utils import sentences_to_ids, split_batch, load_data, build_dict, PAD, DEVICE
from model import Transformer

# 获取项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))

EPOCHS = 20
LEARNING_RATE = 0.0001
RESUME_TRAINING = True

# 文件路径
MODEL_PATH = os.path.join(project_root, 'model.pt')
OPTIMIZER_PATH = os.path.join(project_root, 'optimizer.pt')
TRAINING_STATE_PATH = os.path.join(project_root, 'training_state.json')


def save_training_state(epoch, loss):
    """保存训练状态"""
    state = {
        'epoch': epoch,
        'loss': loss,
        'total_epochs': epoch + 1
    }
    with open(TRAINING_STATE_PATH, 'w') as f:
        json.dump(state, f)


def load_training_state():
    """加载训练状态"""
    if os.path.exists(TRAINING_STATE_PATH):
        with open(TRAINING_STATE_PATH, 'r') as f:
            return json.load(f)
    return {'epoch': 0, 'loss': 0, 'total_epochs': 0}


def train_epoch(model, train_batches, optimizer, criterion):
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
    # 加载训练状态
    training_state = load_training_state()
    start_epoch = training_state['epoch'] if RESUME_TRAINING else 0

    # 加载模型
    model = Transformer(en_total_words, cn_total_words).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    if RESUME_TRAINING and os.path.exists(MODEL_PATH):
        print(f"继续训练，从第 {start_epoch + 1} 轮开始")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        if os.path.exists(OPTIMIZER_PATH):
            optimizer.load_state_dict(torch.load(OPTIMIZER_PATH))
    else:
        print("从头开始训练")
        start_epoch = 0

    for epoch in range(start_epoch, start_epoch + EPOCHS):
        loss = train_epoch(model, train_batches, optimizer, criterion)
        print(f'Epoch {epoch + 1}/{start_epoch + EPOCHS}, Loss: {loss:.4f}')

        # 保存模型和状态
        torch.save(model.state_dict(), MODEL_PATH)
        torch.save(optimizer.state_dict(), OPTIMIZER_PATH)
        save_training_state(epoch + 1, loss)

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