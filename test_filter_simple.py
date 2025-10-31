#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化的正则表达式过滤测试
"""

from data_utils import contains_web_indicators, clean_text

def main():
    print("=== 正则表达式过滤效果测试 ===\n")
    
    # 测试web标识检测
    web_examples = [
        "{{GamesSidebar}}",
        "{{cssxref('align-items')}}",
        "<div class='container'>",
        "function myFunction() {}",
        "https://developer.mozilla.org",
        "See also: Documentation",
        "```javascript\ncode here\n```",
        "background-color: #fff;",
        "console.log('hello')",
        "var myVar = 5;"
    ]
    
    print("1. Web标识检测:")
    for text in web_examples:
        has_web = contains_web_indicators(text)
        print(f"   {has_web:5} | {text}")
    
    print("\n2. 普通文本检测:")
    normal_examples = [
        "Hello, how are you today?",
        "I love learning new languages.",
        "The weather is very nice today.",
        "We should protect the environment.",
        "Education is very important for children."
    ]
    
    for text in normal_examples:
        has_web = contains_web_indicators(text)
        print(f"   {has_web:5} | {text}")
    
    print("\n3. 文本清理:")
    clean_examples = [
        "  Hello   world!  ",
        "...Hello, world!!!",
        "Hello@#$%^&*world"
    ]
    
    for text in clean_examples:
        cleaned = clean_text(text)
        print(f"   原文: '{text}'")
        print(f"   清理: '{cleaned}'")
        print()

if __name__ == "__main__":
    main()
