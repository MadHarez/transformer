import numpy as np
from collections import Counter
import torch
from torch.autograd import Variable

# 全局设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 特殊 token ID
UNK, PAD, BOS, EOS = 0, 1, 2, 3


def load_data(en_path, zh_path):
    """
    加载 Moses 格式平行语料（.en 和 .zh 文件，每行对应一句）。
    :param en_path: 英文文件路径 (e.g., 'data/MDN_Web_Docs-en-zh_CN.en')
    :param zh_path: 中文文件路径 (e.g., 'data/MDN_Web_Docs-en-zh_CN.zh')
    :return: en_sentences, cn_sentences (列表 of 列表)
    """
    with open(en_path, 'r', encoding='utf-8') as en_f:
        en_lines = [line.strip() for line in en_f if line.strip()]  # 跳过空行
    with open(zh_path, 'r', encoding='utf-8') as zh_f:
        zh_lines = [line.strip() for line in zh_f if line.strip()]

    assert len(en_lines) == len(
        zh_lines), f"Files must have the same number of lines: {len(en_lines)} vs {len(zh_lines)}"

    en_sentences = [['BOS'] + line.split() + ['EOS'] for line in en_lines]
    cn_sentences = [['BOS'] + list(line) + ['EOS'] for line in zh_lines]  # 中文按字符分割
    return en_sentences, cn_sentences


def build_dict(sentences, max_words=50000):
    """
    构建词汇表字典。
    :param sentences: 句子列表
    :param max_words: 最大词汇量
    :return: word_dict, total_words, index_dict
    """
    word_count = Counter()
    for sentence in sentences:
        for word in sentence:
            word_count[word] += 1
    most_common = word_count.most_common(max_words)
    total_words = len(most_common) + 4  # + UNK, PAD, BOS, EOS
    word_dict = {w[0]: index + 4 for index, w in enumerate(most_common)}
    word_dict['UNK'] = UNK
    word_dict['PAD'] = PAD
    word_dict['BOS'] = BOS
    word_dict['EOS'] = EOS
    index_dict = {v: k for k, v in word_dict.items()}
    return word_dict, total_words, index_dict


def sentences_to_ids(en_sentences, cn_sentences, en_dict, cn_dict, sort_by_len=True):
    """
    将句子转换为 ID 序列，并可选按长度排序。
    :param en_sentences: 英文句子列表
    :param cn_sentences: 中文句子列表
    :param en_dict: 英文词汇表
    :param cn_dict: 中文词汇表
    :param sort_by_len: 是否按长度排序批次
    :return: out_en_sentences, out_cn_sentences (ID 列表)
    """

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(en_sentences)
        en_sentences = [en_sentences[i] for i in sorted_index]
        cn_sentences = [cn_sentences[i] for i in sorted_index]

    out_en_sentences = [[en_dict.get(w, UNK) for w in sent] for sent in en_sentences]
    out_cn_sentences = [[cn_dict.get(w, UNK) for w in sent] for sent in cn_sentences]
    return out_en_sentences, out_cn_sentences


def seq_padding(X, padding=PAD):
    """
    对序列进行填充。
    :param X: ID 序列列表
    :param padding: 填充值
    :return: 填充后的 numpy 数组
    """
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])


def subsequent_mask(size):
    """
    生成后续掩码（用于 Decoder 自注意力）。
    :param size: 序列长度
    :return: 掩码 tensor
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """
    批次数据类，包含源、目标和掩码。
    """

    def __init__(self, src, trg=None, pad=PAD):
        src = torch.from_numpy(src).to(DEVICE).long()
        trg = torch.from_numpy(trg).to(DEVICE).long()
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def split_batch(en_ids, cn_ids, batch_size=32, shuffle=True):
    """
    分割数据为批次。
    :param en_ids: 英文 ID 列表
    :param cn_ids: 中文 ID 列表
    :param batch_size: 批次大小
    :param shuffle: 是否打乱
    :return: 批次列表 (Batch 对象)
    """
    idx_list = np.arange(0, len(en_ids), batch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    batch_indexs = []
    for idx in idx_list:
        batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en_ids))))
    batches = []
    for batch_index in batch_indexs:
        batch_en = [en_ids[i] for i in batch_index]
        batch_cn = [cn_ids[i] for i in batch_index]
        batch_en = seq_padding(batch_en)
        batch_cn = seq_padding(batch_cn)
        batches.append(Batch(batch_en, batch_cn))
    return batches