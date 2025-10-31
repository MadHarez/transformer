# train.py

import os
import torch
import torch.optim as optim
from torch import nn
import math
import time
from itertools import chain

from data_utils import sentences_to_ids, split_batch, load_data, build_dict, PAD, DEVICE
from model import Transformer

# 获取项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))

EPOCHS = 100  # 继续训练的轮数
LEARNING_RATE = 0.0001
WARMUP_STEPS = 4000  # 预热步数
GRAD_CLIP = 1.0  # 梯度裁剪阈值
LABEL_SMOOTHING = 0.05  # 标签平滑（降低）
PATIENCE = 20  # 早停耐心值（增加）
RESUME_TRAINING = False  # 设置为 True 来继续训练，False 则从头开始
MODEL_PATH = os.path.join(project_root, 'model.pt')
BEST_MODEL_PATH = os.path.join(project_root, 'best_model.pt')
CHECKPOINT_DIR = os.path.join(project_root, 'checkpoints')

D_MODEL = 512  # 模型维度

class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失函数
    """
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clamp(min=1e-9))


class NoamOpt:
    """
    Noam学习率调度器
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._step = 0
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.get_rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def get_rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()


def train_epoch(model, train_batches, optimizer, criterion):
    """
    单 epoch 训练。
    """
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    
    for i, batch in enumerate(train_batches):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        
        if isinstance(criterion, LabelSmoothingLoss):
            loss = criterion(out.contiguous().view(-1, out.size(-1)), batch.trg_y.contiguous().view(-1))
            loss = loss / batch.ntokens
        else:
            loss = criterion(out.contiguous().view(-1, out.size(-1)), batch.trg_y.contiguous().view(-1))
        
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        total_tokens += batch.ntokens
        
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            print(f'Batch {i}/{len(train_batches)}, Loss: {loss.item():.4f}, Tokens/sec: {total_tokens / elapsed:.0f}')
    
    return total_loss / len(train_batches), total_tokens


def evaluate(model, eval_batches, criterion):
    """
    评估模型
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in eval_batches:
            out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            
            if isinstance(criterion, LabelSmoothingLoss):
                loss = criterion(out.contiguous().view(-1, out.size(-1)), batch.trg_y.contiguous().view(-1))
                loss = loss / batch.ntokens
            else:
                loss = criterion(out.contiguous().view(-1, out.size(-1)), batch.trg_y.contiguous().view(-1))
            
            total_loss += loss.item()
            total_tokens += batch.ntokens
    
    return total_loss / len(eval_batches)


def load_existing_model(model_path, en_total_words, cn_total_words):
    """加载已训练的模型"""
    model = Transformer(en_total_words, cn_total_words).to(DEVICE)
    if os.path.exists(model_path):
        print(f"加载已训练模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            return model, checkpoint.get('epoch', 0), checkpoint.get('best_loss', float('inf'))
        else:
            model.load_state_dict(checkpoint)
            return model, 0, float('inf')
    else:
        print("未找到已训练模型，从头开始训练")
        return model, 0, float('inf')


def main_train(en_total_words, cn_total_words, train_batches, val_batches, start_epoch=0, best_loss=float('inf')):
    """
    主训练函数。
    """
    # 加载模型
    if RESUME_TRAINING and os.path.exists(MODEL_PATH):
        model, checkpoint_epoch, checkpoint_best_loss = load_existing_model(MODEL_PATH, en_total_words, cn_total_words)
        start_epoch = max(start_epoch, checkpoint_epoch)
        best_loss = min(best_loss, checkpoint_best_loss)
    else:
        model = Transformer(en_total_words, cn_total_words).to(DEVICE)
    
    # 创建优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    optimizer = NoamOpt(D_MODEL, 1, WARMUP_STEPS, optimizer)
    
    # 使用标签平滑损失
    criterion = LabelSmoothingLoss(cn_total_words, PAD, LABEL_SMOOTHING)
    
    # 早停相关变量
    patience_counter = 0
    
    print(f"开始训练，从epoch {start_epoch}开始，最佳验证损失: {best_loss:.4f}")
    
    for epoch in range(start_epoch, start_epoch + EPOCHS):
        print(f'\n=== Epoch {epoch + 1}/{start_epoch + EPOCHS} ===')
        
        # 训练
        train_loss, train_tokens = train_epoch(model, train_batches, optimizer, criterion)
        
        # 验证
        val_loss = evaluate(model, val_batches, criterion)
        
        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.get_rate():.6f}')
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict(),
                'best_loss': best_loss,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, BEST_MODEL_PATH)
            print(f'保存最佳模型，验证损失: {best_loss:.4f}')
        else:
            patience_counter += 1
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            if not os.path.exists(CHECKPOINT_DIR):
                os.makedirs(CHECKPOINT_DIR)
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict(),
                'best_loss': best_loss,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)
        
        # 保存最新模型
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.optimizer.state_dict(),
            'best_loss': best_loss,
            'train_loss': train_loss,
            'val_loss': val_loss
        }, MODEL_PATH)
        
        # 早停检查
        if patience_counter >= PATIENCE:
            print(f'早停触发，{PATIENCE}个epoch内验证损失没有改善')
            break
    
    print(f'训练完成！最佳验证损失: {best_loss:.4f}')


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

    # 加载数据
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
    train_en_ids, train_cn_ids = sentences_to_ids(train_en, train_cn, en_dict, cn_dict)
    val_en_ids, val_cn_ids = sentences_to_ids(val_en, val_cn, en_dict, cn_dict)
    test_en_ids, test_cn_ids = sentences_to_ids(test_en, test_cn, en_dict, cn_dict)
    
    # 创建批次
    train_batches = split_batch(train_en_ids, train_cn_ids, batch_size=32, shuffle=True)
    val_batches = split_batch(val_en_ids, val_cn_ids, batch_size=32, shuffle=False)
    test_batches = split_batch(test_en_ids, test_cn_ids, batch_size=32, shuffle=False)
    
    print(f"训练批次数: {len(train_batches)}")
    print(f"验证批次数: {len(val_batches)}")
    print(f"测试批次数: {len(test_batches)}")

    # 计算起始epoch（用于显示）
    start_epoch = 0
    best_loss = float('inf')
    if RESUME_TRAINING and os.path.exists(MODEL_PATH):
        # 尝试从检查点加载epoch信息
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            if isinstance(checkpoint, dict):
                start_epoch = checkpoint.get('epoch', 0)
                best_loss = checkpoint.get('best_loss', float('inf'))
        except:
            start_epoch = 0

    main_train(en_total_words, cn_total_words, train_batches, val_batches, start_epoch, best_loss)


if __name__ == "__main__":
    main()