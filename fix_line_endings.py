#!/usr/bin/env python3
import sys

# 读取文件
lines = open('R:/renaissance/workspace/output_sorted.csv', 'r', encoding='utf-8').readlines()

# 写入文件，使用Unix换行符
with open('R:/renaissance/workspace/output_sorted_unix.csv', 'w', encoding='utf-8', newline='\n') as f:
    f.writelines(lines)

print("转换完成！")
