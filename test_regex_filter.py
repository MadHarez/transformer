#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试正则表达式过滤效果
"""

import os
from data_utils import contains_web_indicators, clean_text

def test_regex_filter():
    """测试正则表达式过滤函数"""
    
    # 测试用例
    test_cases = [
        # 应该被过滤的web标识
        ("{{GamesSidebar}}", True, "模板标记"),
        ("{{cssxref('align-items')}}", True, "CSS引用"),
        ("<div class='container'>", True, "HTML标签"),
        ("function myFunction() {}", True, "JavaScript函数"),
        ("https://developer.mozilla.org", True, "URL"),
        ("See also: Documentation", True, "技术文档标识"),
        ("```javascript\ncode here\n```", True, "代码块"),
        ("background-color: #fff;", True, "CSS属性"),
        ("console.log('hello')", True, "console方法"),
        ("var myVar = 5;", True, "变量声明"),
        
        # 应该被保留的普通文本
        ("Hello, how are you today?", False, "日常问候"),
        ("I love learning new languages.", False, "学习相关"),
        ("The weather is very nice today.", False, "天气描述"),
        ("We should protect the environment.", False, "环保话题"),
        ("Education is very important for children.", False, "教育话题"),
        
        # 清理测试
        ("  Hello   world!  ", "Hello world!", "多余空白清理"),
        ("...Hello, world!!!", "Hello, world", "标点清理"),
        ("Hello@#$%^&*world", "Hello world", "特殊字符清理"),
    ]
    
    print("=== 正则表达式过滤测试 ===")
    print()
    
    # 测试web标识检测
    print("1. Web标识检测测试:")
    print("-" * 50)
    for text, expected, description in test_cases[:10]:
        result = contains_web_indicators(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} {description}")
        print(f"   输入: {text}")
        print(f"   预期: {expected}, 实际: {result}")
        print()
    
    # 测试文本清理
    print("2. 文本清理测试:")
    print("-" * 50)
    for text, expected, description in test_cases[10:]:
        result = clean_text(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} {description}")
        print(f"   输入: '{text}'")
        print(f"   输出: '{result}'")
        print(f"   预期: '{expected}'")
        print()
    
    # 测试真实数据样本
    print("3. 真实数据样本测试:")
    print("-" * 50)
    real_samples = [
        "The game loop is advanced by the user's input and sleeps until they provide it.",
        "{{cssxref('align-items')}} CSS property aligns flex items.",
        "function createGameLoop() { return function() { /* loop code */ }; }",
        "Visit https://developer.mozilla.org for more information.",
        "Note: This is an important note about the implementation.",
        "Learning new skills takes time and practice."
    ]
    
    for text in real_samples:
        has_web = contains_web_indicators(text)
        cleaned = clean_text(text)
        print(f"原文: {text}")
        print(f"包含web标识: {has_web}")
        print(f"清理后: {cleaned}")
        print()

if __name__ == "__main__":
    test_regex_filter()
