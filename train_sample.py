#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用高质量示例数据训练翻译模型
"""

import os
import torch
import torch.nn as nn
import math
import time
from itertools import chain

from data_utils import sentences_to_ids, split_batch, build_dict, PAD, DEVICE
from model import Transformer

# 获取项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))

# 训练参数
EPOCHS = 200
LEARNING_RATE = 0.0001
WARMUP_STEPS = 1000
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.05
PATIENCE = 30
BATCH_SIZE = 8  # 小批次，因为数据量小

class LabelSmoothingLoss(nn.Module):
    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = PAD
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
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def train_epoch(model, train_batches, optimizer, criterion):
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
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()
        optimizer.optimizer.zero_grad()  # 修复：调用内部优化器的zero_grad
        
        total_loss += loss.item()
        total_tokens += batch.ntokens
        
        if i % 10 == 0 and i > 0:
            elapsed = time.time() - start_time
            print(f'Batch {i}/{len(train_batches)}, Loss: {loss.item():.4f}, Tokens/sec: {total_tokens / elapsed:.0f}')
    
    return total_loss / len(train_batches), total_tokens

def evaluate(model, eval_batches, criterion):
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

def main():
    print("使用高质量示例数据训练翻译模型...")
    
    # 加载示例数据
    project_root = os.path.dirname(os.path.abspath(__file__))
    en_path = os.path.join(project_root, 'data', 'sample.en')
    zh_path = os.path.join(project_root, 'data', 'sample.zh')
    
    if not os.path.exists(en_path) or not os.path.exists(zh_path):
        print("请先运行 create_sample_data.py 创建示例数据")
        return
    
    # 读取数据
    with open(en_path, 'r', encoding='utf-8') as f:
        en_lines = [line.strip() for line in f if line.strip()]
    with open(zh_path, 'r', encoding='utf-8') as f:
        zh_lines = [line.strip() for line in f if line.strip()]
    
    print(f"加载数据: {len(en_lines)}对句子")
    
    # 构建词汇表和句子
    en_sentences = [['BOS'] + line.split() + ['EOS'] for line in en_lines]
    cn_sentences = [['BOS'] + list(line) + ['EOS'] for line in zh_lines]
    
    # 构建字典
    en_dict, en_total_words, en_index_dict = build_dict(en_sentences)
    cn_dict, cn_total_words, cn_index_dict = build_dict(cn_sentences)
    
    print(f"英文词汇表大小: {en_total_words}")
    print(f"中文词汇表大小: {cn_total_words}")
    
    # 转换为ID
    en_ids, cn_ids = sentences_to_ids(en_sentences, cn_sentences, en_dict, cn_dict)
    
    # 数据分割
    total_size = len(en_ids)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    
    train_en = en_ids[:train_size]
    train_cn = cn_ids[:train_size]
    val_en = en_ids[train_size:train_size+val_size]
    val_cn = cn_ids[train_size:train_size+val_size]
    test_en = en_ids[train_size+val_size:]
    test_cn = cn_ids[train_size+val_size:]
    
    print(f"训练集: {len(train_en)}对")
    print(f"验证集: {len(val_en)}对")
    print(f"测试集: {len(test_en)}对")
    
    # 创建批次
    train_batches = split_batch(train_en, train_cn, BATCH_SIZE, shuffle=True)
    val_batches = split_batch(val_en, val_cn, BATCH_SIZE, shuffle=False)
    test_batches = split_batch(test_en, test_cn, BATCH_SIZE, shuffle=False)
    
    print(f"训练批次数: {len(train_batches)}")
    print(f"验证批次数: {len(val_batches)}")
    print(f"测试批次数: {len(test_batches)}")
    
    # 创建模型
    model = Transformer(en_total_words, cn_total_words).to(DEVICE)
    optimizer = get_std_opt(model)
    criterion = LabelSmoothingLoss(cn_total_words, LABEL_SMOOTHING)
    
    print("开始训练...")
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\\n=== Epoch {epoch}/{EPOCHS} ===")
        
        # 训练
        train_loss, train_tokens = train_epoch(model, train_batches, optimizer, criterion)
        
        # 验证
        val_loss = evaluate(model, val_batches, criterion)
        
        current_lr = optimizer.rate()
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_path = os.path.join(project_root, 'best_sample_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict(),  # 修复：保存内部优化器状态
                'best_loss': best_loss,
                'en_dict': en_dict,
                'cn_dict': cn_dict,
                'en_total_words': en_total_words,
                'cn_total_words': cn_total_words
            }, best_model_path)
            print(f"保存最佳模型，验证损失: {val_loss:.4f}")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= PATIENCE:
            print(f"早停触发，{PATIENCE}个epoch内验证损失没有改善")
            break
    
    print(f"训练完成！最佳验证损失: {best_loss:.4f}")
    
    # 测试最佳模型
    best_model_path = os.path.join(project_root, 'best_sample_model.pt')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss = evaluate(model, test_batches, criterion)
        test_perplexity = math.exp(test_loss)
        
        print(f"\\n=== 测试结果 ===")
        print(f"测试损失: {test_loss:.4f}")
        print(f"测试困惑度: {test_perplexity:.4f}")

if __name__ == "__main__":
    from torch.autograd import Variable
    main()
