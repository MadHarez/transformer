import numpy as np
from collections import Counter
import torch
import re
from torch.autograd import Variable

# 全局设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 特殊 token ID
UNK, PAD, BOS, EOS = 0, 1, 2, 3


def contains_web_indicators(text):
    """
    使用正则表达式检测文本中的web标识和技术标记
    """
    # Web模板标记
    template_patterns = [
        r'\{\{[^}]*\}\}',  # {{template}}
        r'\{\{[^}]*$',      # 不完整的模板开始
        r'\}\}',            # 模板结束
    ]
    
    # HTML标签和属性
    html_patterns = [
        r'<[^>]+>',          # HTML标签
        r'&[a-zA-Z]+;',      # HTML实体
        r'\w+[\-_]*\w*=\[', # 属性赋值 [attr=value]
        r'\w+[\-_]*\w*="[^"]*"', # 属性赋值 attr="value"
    ]
    
    # CSS相关
    css_patterns = [
        r'cssxref\([^)]*\)', # CSS引用
        r'css-[\w-]+',       # CSS类名
        r'[\.#][\w-]+\s*\{', # CSS选择器
        r'background|color|font|margin|padding', # CSS属性
    ]
    
    # JavaScript相关
    js_patterns = [
        r'function\s+\w+\s*\(',  # 函数定义
        r'\b(var|let|const)\s+\w+', # 变量声明
        r'\w+\s*=>\s*',           # 箭头函数
        r'\w+\.\w+',              # 对象方法调用
        r'console\.\w+',           # console方法
    ]
    
    # URL和文件路径
    url_patterns = [
        r'https?://[^\s]+',    # HTTP/HTTPS URL
        r'www\.[^\s]+',        # www URL
        r'[\w\-\.]+\.[a-zA-Z]{2,}', # 域名
        r'/[\w\-\/]+',        # 文件路径
        r'[\w\-]+\.[\w\-]+', # 文件名
    ]
    
    # 技术文档标识
    doc_patterns = [
        r'\bNote:\b',
        r'\bSee also:\b',
        r'\bSyntax:\b',
        r'\bParameters?:\b',
        r'\bReturn value:\b',
        r'\bExample:\b',
        r'\bWarning:\b',
        r'\bImportant:\b',
        r'\bTODO:\b',
        r'\bFIXME:\b',
    ]
    
    # 代码块和代码标识
    code_patterns = [
        r'```[^`]*```',         # 代码块
        r'`[^`]+`',             # 行内代码
        r'/\*[^*]*\*/',        # 多行注释
        r'//[^\n]*',           # 单行注释
        r'[{}();]',             # 编程语言符号
        r'\b(if|else|for|while|do|switch|case|break|continue|return)\b', # 关键字
    ]
    
    # 数字和特殊符号过多
    technical_patterns = [
        r'\d+\.\d+',           # 小数
        r'\b\d{2,}\b',         # 两位以上数字
        r'[\[\]()]',           # 方括号圆括号
        r'[|&<>]',              # 逻辑和比较符号
    ]
    
    # 合并所有模式
    all_patterns = (
        template_patterns + html_patterns + css_patterns + 
        js_patterns + url_patterns + doc_patterns + 
        code_patterns + technical_patterns
    )
    
    for pattern in all_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False

def clean_text(text):
    """
    清理文本中的轻微噪声
    """
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 移除开头和结尾的标点
    text = text.strip('.,;:!?"\'()[]{}')
    
    # 移除特殊字符但保留基本标点
    text = re.sub(r'[^\w\s\u4e00-\u9fff.,;:!?"\'()-]', ' ', text)
    
    # 再次清理空白
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text

def load_data(en_path, zh_path, min_length=5, max_length=100):
    """
    加载 Moses 格式平行语料（.en 和 .zh 文件，每行对应一句）。
    使用正则表达式过滤低质量数据和web标识。
    :param en_path: 英文文件路径
    :param zh_path: 中文文件路径
    :param min_length: 最小句子长度
    :param max_length: 最大句子长度
    :return: en_sentences, cn_sentences (列表 of 列表)
    """
    with open(en_path, 'r', encoding='utf-8') as en_f:
        en_lines = [line.strip() for line in en_f if line.strip()]
    with open(zh_path, 'r', encoding='utf-8') as zh_f:
        zh_lines = [line.strip() for line in zh_f if line.strip()]

    assert len(en_lines) == len(
        zh_lines), f"Files must have the same number of lines: {len(en_lines)} vs {len(zh_lines)}"

    # 过滤数据
    filtered_en = []
    filtered_zh = []
    
    print("开始使用正则表达式过滤数据...")
    
    for i, (en_line, zh_line) in enumerate(zip(en_lines, zh_lines)):
        # 使用正则表达式检测web标识
        if contains_web_indicators(en_line) or contains_web_indicators(zh_line):
            continue
        
        # 清理文本
        en_clean = clean_text(en_line)
        zh_clean = clean_text(zh_line)
        
        # 清理后再次检查
        if contains_web_indicators(en_clean) or contains_web_indicators(zh_clean):
            continue
        
        # 跳过空行
        if not en_clean or not zh_clean:
            continue
            
        # 跳过过短或过长的句子
        en_words = en_clean.split()
        zh_chars = list(zh_clean)
        if len(en_words) < min_length or len(en_words) > max_length:
            continue
        if len(zh_chars) < min_length or len(zh_chars) > max_length:
            continue
        
        # 跳过包含过多标点符号的行
        en_punct_ratio = sum(1 for c in en_clean if not c.isalnum() and not c.isspace()) / len(en_clean) if en_clean else 0
        zh_punct_ratio = sum(1 for c in zh_clean if not c.isalnum() and not c.isspace() and ord(c) < 128) / len(zh_clean) if zh_clean else 0
        if en_punct_ratio > 0.3 or zh_punct_ratio > 0.3:
            continue
        
        # 跳过包含大量数字的行
        en_digit_ratio = sum(1 for c in en_clean if c.isdigit()) / len(en_clean) if en_clean else 0
        if en_digit_ratio > 0.1:
            continue
        
        filtered_en.append(en_clean)
        filtered_zh.append(zh_clean)
        
        # 每1000行显示进度
        if (i + 1) % 1000 == 0:
            print(f"已处理 {i + 1}/{len(en_lines)} 行，保留 {len(filtered_en)} 对")
    
    print(f"原始数据: {len(en_lines)}对，正则过滤后: {len(filtered_en)}对")
    
    en_sentences = [['BOS'] + line.split() + ['EOS'] for line in filtered_en]
    cn_sentences = [['BOS'] + list(line) + ['EOS'] for line in filtered_zh]  # 中文按字符分割
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