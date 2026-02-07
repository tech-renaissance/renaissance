#!/usr/bin/env python3
import sys

# 读取原始文件
with open('R:/renaissance/workspace/output.csv', 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f]

# 排序
sorted_lines = sorted(lines)

# 写入文件，使用Unix换行符
with open('R:/renaissance/workspace/output_sorted.csv', 'w', encoding='utf-8', newline='\n') as f:
    for line in sorted_lines:
        f.write(line + '\n')

print(f"处理完成！共{len(sorted_lines)}行")
