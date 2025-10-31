#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from data_utils import load_data

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    en_path = os.path.join(project_root, 'data', 'MDN_Web_Docs.en-zh_CN.en')
    zh_path = os.path.join(project_root, 'data', 'MDN_Web_Docs.en-zh_CN.zh_CN')

    print("正在检查过滤后的数据质量...")
    en_sentences, cn_sentences = load_data(en_path, zh_path)

    print(f'\n过滤后的数据样本（前10个）：')
    print('=' * 60)
    
    for i in range(min(10, len(en_sentences))):
        en = ' '.join(en_sentences[i][1:-1])  # 去掉BOS和EOS
        zh = ''.join(cn_sentences[i][1:-1])   # 去掉BOS和EOS
        print(f'{i+1}. EN: {en}')
        print(f'   ZH: {zh}')
        print()

if __name__ == "__main__":
    main()
