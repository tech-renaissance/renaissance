# BOM、编码与换行符统一方案 —— 最终版（修订）

> 综合 EKT1 ~ EKT4 四套方案及 REV.md 小伙伴意见，形成最终执行方案。

---

## 一、调研结论（综合四份报告）

### 1.1 代码文件（.h / .hpp / .c / .cpp / .cu / .cuh）

| 指标 | 数值 | 来源 |
|------|------|------|
| 扫描文件总数 | 337 | EKT2/EKT4 |
| UTF-8 无 BOM | 334（99.1%） | EKT2/EKT4 |
| UTF-8 带 BOM | 3（0.9%） | 全部一致 |
| LF 换行 | 295（87.5%） | 全部一致 |
| CRLF 换行 | 42（12.5%） | 全部一致 |
| mixed 换行 | 0 | 全部一致 |
| 文件末尾缺少换行符 | 152 | EKT3 |

**带 BOM 的 3 个文件（全部一致）：**

- `include/renaissance/core/logger.h`
- `src/core/logger.cpp`
- `src/data/random_crop.cpp`

**Git 索引关键发现（EKT4）：**

- `git ls-files --eol` 显示 **Git 索引（仓库实际存储）几乎全部为 LF**。
- 工作区出现 CRLF 是因为全局 `core.autocrlf=true` 在 checkout 时自动转换。
- 也就是说，**仓库历史本来就是 LF，CRLF 只是工作区污染**，统一为 LF 本质上是"恢复"而非"改变"。

### 1.2 非代码文本文件（.md / .py / .json / .cmake / .txt / .bat 等）

| 指标 | 数值 | 来源 |
|------|------|------|
| 扫描文件总数 | ~163 | EKT4 |
| LF | ~116 | EKT4 |
| CRLF | ~45 | EKT4 |
| MIXED | 1（`docs/TANH_TEST_D.md`） | EKT4 |
| UTF-8 带 BOM | 1（`docs/TANH_TEST_D.md`） | EKT4 |
| UTF-16-LE | 1（`tests/correction/pt_gpu_temp.txt`） | EKT4 |

### 1.3 特殊异常（EKT4）

| 文件 | 问题 | 严重程度 | 处理策略 |
|------|------|----------|----------|
| `setup_cuda_env.bat` | 换行符为 `\r\r\n`（双 CR），非标准 CRLF | 高 | 修复为标准 CRLF |
| `docs/TANH_TEST_D.md` | BOM + MIXED_CRLF_LF | 中 | 去 BOM + 统一为 LF |
| `tests/correction/pt_gpu_temp.txt` | UTF-16-LE 编码 | 待确认 | 需先调查用途，见下文 2.4 |

> 注意：`build.bat` 和 `.gitignore` 在工作区处于已删除状态（`D`），属于仓库管理问题，不纳入本方案。执行前应单独确认其存在性。

### 1.4 项目配置现状（全部一致）

- `.gitattributes`：**不存在**
- `.editorconfig`：**不存在**
- `core.autocrlf`：`true`（导致工作区 CRLF 污染的根因）

---

## 二、最终统一方案

### 2.1 编码：UTF-8 without BOM

**对所有文本文件统一为 UTF-8 无 BOM。**

需处理：

| 文件 | 当前 | 目标 |
|------|------|------|
| `include/renaissance/core/logger.h` | UTF-8 BOM | UTF-8 无 BOM |
| `src/core/logger.cpp` | UTF-8 BOM | UTF-8 无 BOM |
| `src/data/random_crop.cpp` | UTF-8 BOM | UTF-8 无 BOM |
| `docs/TANH_TEST_D.md` | UTF-8 BOM | UTF-8 无 BOM |
| `tests/correction/pt_gpu_temp.txt` | UTF-16-LE | 待调查后决定 |

### 2.2 换行符：LF

**对所有文本文件统一为 LF，三个例外：**

| 例外 | 换行符 | 原因 |
|------|--------|------|
| `*.bat` | CRLF | Windows 批处理文件硬性要求 |
| `*.cmd` | CRLF | 同上 |
| 第三方代码 | 保持原样 | 便于同步上游（`node.hpp`、`fkyaml_fwd.hpp`、`Simd/`） |

**需 CRLF→LF 转换的文件：**

- 代码文件：42 个（清单见 EKT2 第 4.5 节）
- 非代码文本文件：约 45 个
- Mixed 修复：`docs/TANH_TEST_D.md` → LF

### 2.3 文件结尾换行符

**所有文本文件必须以单个换行符（`\n`）结尾。**

这是 POSIX 标准的定义（a line is a sequence of zero or more non-newline characters plus a terminating newline character），也是绝大多数开源项目的惯例。EKT3 发现 152 个代码文件缺少结尾换行符，需统一补充。

> 注意：此处指文件末尾恰好一个 `\n`，**不是**额外多一个空行（即末尾不是 `\n\n`）。

补充结尾换行符不会改变编译产物：C/C++ 标准翻译阶段 2 中，文件末尾的反斜杠-换行符拼接仅影响行续接，末尾独立的 `\n` 不产生任何 token。

### 2.4 特殊异常修复

| 文件 | 修复方案 |
|------|----------|
| `setup_cuda_env.bat` | 将 `\r\r\n` 修复为标准 CRLF（`\r\n`） |
| `docs/TANH_TEST_D.md` | 去 BOM + 统一为 LF |
| `tests/correction/pt_gpu_temp.txt` | **先调查用途**：查看该文件在测试代码中是被作为文本读取还是按 UTF-16 二进制比对。若为二进制参考数据则保持原样并标记为 binary；若为人工文本则转码为 UTF-8 无 BOM + LF |

---

## 三、强制执行机制

### 3.1 `.gitattributes`（仓库级强制）

在仓库根目录新增 `.gitattributes`。采用显式声明策略，不使用 `* text=auto` 兜底，避免未列出的二进制文件被误判为文本并损坏。

```gitattributes
# ============================================================
# 第三方代码：不参与任何换行符/编码归一化，保持原样
# ============================================================
include/renaissance/core/node.hpp       -text
include/renaissance/core/fkyaml_fwd.hpp -text
Simd/**                                 -text

# ============================================================
# 生成代码：由生成器决定，不参与归一化
# ============================================================
include/generated/**                    -text

# ============================================================
# 源代码：统一为 LF
# ============================================================
*.cpp   text eol=lf
*.h     text eol=lf
*.hpp   text eol=lf
*.cu    text eol=lf
*.cuh   text eol=lf
*.c     text eol=lf

# ============================================================
# 脚本与文档：统一为 LF
# ============================================================
*.py      text eol=lf
*.sh      text eol=lf
*.md      text eol=lf
*.txt     text eol=lf
*.json    text eol=lf
*.yaml    text eol=lf
*.yml     text eol=lf
*.cmake   text eol=lf
*.cfg     text eol=lf
*.ini     text eol=lf
CMakeLists.txt text eol=lf

# ============================================================
# Windows 批处理：保留 CRLF
# ============================================================
*.bat     text eol=crlf
*.cmd     text eol=crlf

# ============================================================
# 二进制文件：禁止转换
# ============================================================
# 图片
*.png     binary
*.jpg     binary
*.jpeg    binary
*.gif     binary
*.ico     binary
*.bmp     binary
*.svg     binary

# 文档
*.pdf     binary

# 压缩包
*.zip     binary
*.tar     binary
*.gz      binary
*.bz2     binary
*.7z      binary

# 库文件
*.dll     binary
*.exe     binary
*.so      binary
*.a       binary
*.lib     binary

# Python 编译产物
*.pyc     binary
*.pyo     binary

# 模型/权重文件
*.onnx    binary
*.pth     binary
*.pt      binary
*.weights binary
*.pb      binary

# 音视频
*.mp4     binary
*.wav     binary
*.mp3     binary

# 字体
*.ttf     binary
*.woff    binary
*.woff2   binary
```

> 注意：`.gitattributes` 文件自身也应以 UTF-8 无 BOM + LF 格式保存。

### 3.2 `.editorconfig`（编辑器级辅助）

新增 `.editorconfig`，让主流 IDE/编辑器在打开文件时自动应用正确设置。

```ini
root = true

# ============================================================
# 全局默认
# ============================================================
[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true

# ============================================================
# C/C++ 源代码
# ============================================================
[*.{cpp,h,hpp,cu,cuh,c}]
indent_style = space
indent_size = 4

# ============================================================
# Markdown：关闭行尾空格修剪（两个 trailing space 表示硬换行）
# ============================================================
[*.md]
trim_trailing_whitespace = false

# ============================================================
# JSON / YAML：可安全修剪行尾空格
# ============================================================
[*.{json,yaml,yml}]
indent_style = space
indent_size = 2
trim_trailing_whitespace = true

# ============================================================
# Windows 批处理
# ============================================================
[*.bat]
end_of_line = crlf

[*.cmd]
end_of_line = crlf
```

**设计说明：**

- 全局**不启用** `trim_trailing_whitespace`，避免破坏 C/C++ 字符串字面量中可能有意义的行尾空格（如多行字符串拼接）。
- 仅对 `.json`、`.yaml`、`.yml` 等不存在此问题的文件类型启用行尾空格修剪。
- 缩进规则限定在 C/C++ 源代码（4 空格），不强制其他文件类型。
- 全局 `insert_final_newline = true` 确保编辑器保存时自动补结尾换行符。

### 3.3 两者的分工

| 机制 | 作用层面 | 生效时机 |
|------|----------|----------|
| `.gitattributes` | 仓库级强制 | `git add` / `git checkout` 时 |
| `.editorconfig` | 编辑器级辅助 | 打开/保存文件时 |

`.gitattributes` 是**底线**——不管开发者用什么编辑器、什么 OS，提交到仓库的一定是 LF。`.editorconfig` 是**便利**——让开发者在编辑时就看到正确的格式，减少心智负担。

---

## 四、执行计划

### 4.1 前置条件

1. 当前 CMD.md 注释规范化批次已执行完毕。注意：该批次因在大量文件头部插入多行注释导致 `__LINE__` 宏值偏移，约 138 个构建产物（`.exe`、`.lib` 及部分 `.obj`）哈希与 `build_legacy/` 不同。**此差异已判定为行号偏移导致，与代码逻辑无关，属于可接受的产物差异。**
2. 确认 `build_legacy/` 目录完整可用，作为后续哈希对比的基准。
3. 确认 `build.bat` 和 `.gitignore` 的存在性（当前工作区显示为已删除，需先解决）。

### 4.2 执行步骤

**第 1 步：新建独立分支**

```
git checkout -b chore/encoding-line-ending-unification
```

独立分支便于 review、回滚，不与功能分支混淆。

**第 2 步：新增配置文件**

提交 `.gitattributes` 和 `.editorconfig`（内容见第三部分）。

> 必须在 `git add --renormalize` 之前添加 `.gitattributes`，否则 Git 无法按项目规则执行归一化。

**第 3 步：调查 `pt_gpu_temp.txt` 的用途**

在 `tests/` 中搜索对该文件的引用：

```
grep -r "pt_gpu_temp" tests/
```

- 若该文件被测试代码以 UTF-16 二进制方式读取并比对，则**保持原样**，并在 `.gitattributes` 中追加 `tests/correction/pt_gpu_temp.txt binary`。
- 若为人工编写的参考文本，则转码为 UTF-8 无 BOM + LF。

**第 4 步：处理异常文件**

| 优先级 | 操作 | 文件 |
|--------|------|------|
| 1 | 修复 `\r\r\n` → 标准 CRLF | `setup_cuda_env.bat` |
| 2 | 去 BOM + 统一为 LF | `docs/TANH_TEST_D.md` |

**第 5 步：执行换行符归一化**

```
git add --renormalize .
```

> `--renormalize` 的作用：对已跟踪的文件，按 `.gitattributes` 规则重新应用换行符转换。它会自动完成 CRLF → LF 转换（对声明了 `eol=lf` 的文件），但**不会**改变 BOM、**不会**补充文件结尾换行符。BOM 和结尾换行符需单独处理。

**第 6 步：去除 BOM**

对 3 个代码文件 + `docs/TANH_TEST_D.md`（若第 4 步未处理）去除 BOM 头（`EF BB BF` 三字节）。使用 Python 脚本逐文件处理，不批量替换。

**第 7 步：补充文件结尾换行符**

对 152 个缺少结尾换行符的代码文件及非代码文本文件，逐个补充 `\n`。使用脚本检测并修复，人工抽检确保仅追加换行符而未改动文件内容。

**第 8 步：验证编译产物哈希**

全量编译，对比 `build/` 与 `build_legacy/` 产物哈希。C/C++ 编译器对换行符和 BOM 不敏感，理论上应完全一致（除前置条件中已明确的 138 个行号偏移产物外）。

**第 9 步：更新 CMD.md**

将第 8.3 节：
> 本次不修改文件编码（UTF-8 BOM）和换行符（CRLF/LF），维持现状。换行符统一问题留待后续处理。

替换为：
> 文件编码统一为 UTF-8 without BOM，换行符统一为 LF（`.bat`/`.cmd` 除外，使用 CRLF）。详见 `.gitattributes` 和 `.editorconfig`。

**第 10 步：CI 校验（可选，建议后续跟进）**

在 CI 中加入检查脚本，防止问题回潮。建议使用 Python 脚本以保证跨平台兼容：

```python
#!/usr/bin/env python3
"""CI 检查：编码与换行符规范。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDE = {"node.hpp", "fkyaml_fwd.hpp"}
EXCLUDE_DIRS = {"Simd", "generated"}

def check_file(path: Path) -> list[str]:
    errors = []
    raw = path.read_bytes()
    # 检查 BOM
    if raw.startswith(b"\xef\xbb\xbf"):
        errors.append(f"  BOM found: {path}")
    # 检查 CRLF（.bat/.cmd 除外）
    if path.suffix not in (".bat", ".cmd") and b"\r\n" in raw:
        errors.append(f"  CRLF found: {path}")
    # 检查结尾换行符
    if raw and not raw.endswith(b"\n"):
        errors.append(f"  missing final newline: {path}")
    return errors

all_errors = []
for ext in ("*.h", "*.hpp", "*.cpp", "*.cu", "*.cuh", "*.c", "*.py", "*.md", "*.txt", "*.json", "*.cmake"):
    for path in ROOT.rglob(ext):
        if any(p in path.parts for p in EXCLUDE_DIRS):
            continue
        if path.name in EXCLUDE:
            continue
        all_errors.extend(check_file(path))

if all_errors:
    for e in all_errors:
        print(e, file=sys.stderr)
    sys.exit(1)
print("All files pass encoding/line-ending checks.")
```

### 4.3 验证方法

| 验证项 | 方法 |
|--------|------|
| 编码正确 | 检查文件头是否无 `EF BB BF`；`pt_gpu_temp.txt` 若保留则确认其编码不变 |
| 换行符正确 | `git ls-files --eol`：`.cpp`/`.h`/`.cu` 等应显示 `w/lf` 且 `attr/` 列为 `text eol=lf`；`.bat`/`.cmd` 应显示 `w/crlf` 且 `attr/` 列为 `text eol=crlf` |
| 第三方代码未改动 | `git diff --stat` 确保 `node.hpp`、`fkyaml_fwd.hpp`、`Simd/` 无变更 |
| 编译产物一致 | `build/` vs `build_legacy/` 哈希对比（排除已知的 138 个行号偏移产物） |
| 结尾换行符 | `git diff --check` 不应报 "no newline at end of file" |

---

## 五、影响评估

### 5.1 编译产物

| 改动 | 对编译产物的影响 |
|------|------------------|
| 去 BOM | 无影响。BOM 在翻译阶段 1 被处理 |
| CRLF→LF | 无影响。C/C++ 标准翻译阶段 1 将 `\r\n` 和 `\n` 统一映射为换行符 |
| 补充结尾换行符 | 无影响。文件末尾的 `\n` 不产生任何 token |
| 修复 `\r\r\n` | 无影响（`.bat` 不参与编译） |
| `pt_gpu_temp.txt` 转码 | 若保留原样则无影响；若转码需确认测试逻辑不受影响 |

**结论：所有改动均不改变编译产物，符合"不改变编译出来的二进制文件"的核心原则。**

### 5.2 注释乱码风险

| 风险 | 分析 |
|------|------|
| 去 BOM 后中文注释乱码 | 无风险。BOM 仅标识编码，去除后文件内容不变，UTF-8 中文注释完整保留 |
| CRLF→LF 后中文乱码 | 无风险。换行符转换不涉及多字节字符，UTF-8 中文不受影响 |
| `trim_trailing_whitespace` | 已全局禁用，仅对 `.json`/`.yaml` 等安全类型启用，不会触碰 C/C++ 源码 |

### 5.3 Git 历史

- 一次性全量修改会产生大量 diff。应在**独立分支**和**独立 commit** 中完成，commit message 标注 `chore: normalize encoding and line endings`。
- 后续代码审查可使用 `git diff -w`（忽略空白）或 `git blame --ignore-rev <commit>` 跳过此 commit。

### 5.4 开发者体验

- **Windows 开发者**：现代 IDE（VS 2022、VS Code、CLion、Rider）均原生支持 LF，无需额外配置。`.editorconfig` 提供自动设置。
- **Linux/macOS 开发者**：LF 是原生换行符，无任何影响。
- **`.bat` 文件**：保持 CRLF，Windows 开发者可直接运行。

### 5.5 风险矩阵

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| 编译产物变化 | 无 | C/C++ 标准保证等价 |
| 中文注释乱码 | 无 | 换行符和 BOM 操作不涉及多字节字符 |
| `.bat` 文件损坏 | 低 | 明确排除，保持 CRLF；修复 `\r\r\n` 异常 |
| 第三方代码被篡改 | 无 | `.gitattributes` 显式 `-text` 排除 |
| Git diff 噪音 | 低 | 独立分支 + 独立 commit |
| `pt_gpu_temp.txt` 转码破坏测试 | 中 | 先调查用途，再决定处理方式 |
| `.gitattributes` 自身格式错误 | 低 | 保存时人工确认 UTF-8 无 BOM + LF |

---

## 六、方案对比总结

| 维度 | EKT1 | EKT2 | EKT3 | EKT4 | **EKT_FINAL（修订）** |
|------|------|------|------|------|------------------------|
| 编码统一为 UTF-8 无 BOM | ✓ | ✓ | ✓ | ✓ | ✓ |
| 换行符统一为 LF | ✓ | ✓ | ✓ | ✓ | ✓ |
| `.bat` 保持 CRLF | ✓ | — | ✓ | ✓ | ✓ |
| `.gitattributes` | ✓ | ✓ | ✓ | ✓ | ✓（显式排除第三方+生成代码，无 `* text=auto` 兜底） |
| `.editorconfig` | — | — | ✓ | — | ✓（按文件类型分控，全局不 trim） |
| 文件以 `\n` 结尾 | — | — | ✓ | — | ✓（明确表述，独立步骤） |
| 非代码文本文件扫描 | ✓ | — | — | ✓ | ✓ |
| Git 索引分析 | — | — | — | ✓ | ✓ |
| 特殊异常修复 | — | — | — | ✓ | ✓（`\r\r\n`；`pt_gpu_temp.txt` 先调查） |
| CI 校验建议 | — | ✓ | — | — | ✓（跨平台 Python 脚本） |
| 可复现调研脚本 | — | ✓ | — | ✓ | 引用 EKT2 脚本 |
| 实施步骤 | ✓ | ✓ | ✓ | ✓ | ✓（10 步，顺序修正） |
| 风险分析 | ✓ | — | ✓ | — | ✓（含乱码风险评估） |
| 前置条件说明 | — | — | — | — | ✓（说明 138 个行号偏移产物） |

---

## 七、最终结论

本项目开源前的编码与换行符统一方案为：

> **UTF-8 without BOM + LF 换行符 + 文件以单个换行符（`\n`）结尾**

通过 `.gitattributes`（仓库级强制，显式排除第三方和生成代码）和 `.editorconfig`（编辑器级辅助，按文件类型分控）双机制长期维持。例外仅两项：Windows 批处理文件（`.bat`/`.cmd`）保持 CRLF；第三方代码（`node.hpp`、`fkyaml_fwd.hpp`、`Simd/`）保持原样。

执行窗口为当前 CMD.md 注释规范化批次验收之后，在独立分支中一次性完成。改动不改变任何编译产物，不引入中文注释乱码，符合项目核心原则。