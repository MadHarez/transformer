# infer.py
import torch
from torch.autograd import Variable

from data_utils import PAD, UNK, BOS, DEVICE, EOS, subsequent_mask
from model import MAX_LENGTH


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    贪婪解码用于推理。
    :param model: 训练好的模型
    :param src: 输入源 tensor
    :param src_mask: 源掩码
    :param max_len: 最大输出长度
    :param start_symbol: BOS
    :return: 输出 ID tensor
    """
    model.eval()
    memory = model.encoder(model.src_embed(src), src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).to(DEVICE)
    for i in range(max_len - 1):
        out = model.decoder(model.tgt_embed(ys), memory, src_mask, Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def translate_sentence(model, sentence, en_dict, cn_index_dict, max_len=MAX_LENGTH):
    """
    翻译单句。
    :param model: 模型
    :param sentence: 英文字符串
    :param en_dict: 英文词汇表
    :param cn_index_dict: 中文索引到词
    :param max_len: 最大长度
    :return: 翻译字符串
    """
    src = torch.LongTensor([[en_dict.get(w, UNK) for w in ['BOS'] + sentence.split() + ['EOS']]]).to(DEVICE)
    src_mask = (src != PAD).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask, max_len, BOS)
    translation = ''.join([cn_index_dict.get(int(i), 'UNK') for i in out[0] if int(i) not in [BOS, EOS, PAD]])
    return translation

# 示例使用（需加载词汇表）
# model = Transformer(en_total_words, cn_total_words).to(DEVICE)
# model.load_state_dict(torch.load('model.pt'))
# print(translate_sentence(model, "Hello world", en_dict, cn_index_dict))