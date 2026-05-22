# 去重指南

## 一、问题

6 变体 × 17 GraphId = 102 个逻辑槽位。若逐一捕获，102 次 `cudaGraphInstantiate` 耗时数十秒，且大半是冗余——8 种 shape 无关图（TRANSFER/COMM/ZERO_GRAD/CAST_AND_CHECK/EMA）在 6 个变体下完全相同，却被重复捕获 48 次。

目标：**只对真正不同的图捕获一次，其余槽位指向同一个 `CapturedGraph`**。

## 二、去重键

`CapturedGraph::Key` 三元组决定图的唯一身份（[captured_graph.h:L117](file:///r:\renaissance\include\renaissance\graph\captured_graph.h#L117)）：

```cpp
struct Key {
    const ComputationGraph* cg;   // 拓扑：同一 CG 对象指针
    GraphId                 gid;  // 子图：DEEP_FWD_BWD / OPTIMIZER / ...
    ShapeId                 shape; // 输入形状：(N,H,W,C) 四元组，或 kShapeInvariant
};
```

三个维度全部相等才算同一张图。例如：同一 CG 对象的 `DEEP_FWD_BWD` + `ShapeId{128,256,256,3}` 和 `DEEP_FWD_BWD` + `ShapeId{128,128,128,3}` 是两个不同的 Key → 需要两张不同的图。

**为什么用 ShapeId 四元组而非 MemoryPlan 指针？**
不同变体的 MemoryPlan 是不同对象，指针必然不同，即使 shape 相同也无法碰撞。ShapeId 比较的是值——相同输入尺寸 → 相同 ShapeId → Key 相等 → 自动复用。这是"按语义匹配而非按地址匹配"的去重。

## 三、shape 无关图的全变体复用

[is_shape_invariant_graph()](file:///r:\renaissance\include\renaissance\graph\computation_graph.h#L127) 标记 8 种 GraphId 不依赖输入尺寸：

```
TRANSFER_A, TRANSFER_B, ZERO_GRAD, CAST_AND_CHECK,
FIRST_COMM, DEEP_COMM, STATS_COMM, EMA_UPDATE
```

Phase A 编译时，Compiler 给这些槽位统一填入 `kShapeInvariant{0,0,0,0}`（[shape_id.h:L73](file:///r:\renaissance\include\renaissance\graph\shape_id.h#L73)）。Phase B 遍历时，6 个变体的同一种 GraphId 全部产生相同的 Key `{cg, TRANSFER_A, {0,0,0,0}}` → 首次未命中（捕获），后续 5 次全部命中（复用）→ 8 张图服务 48 个槽位。

## 四、去重算法

[pre_capture()](file:///r:\renaissance\src\graph\captured_graph.cpp#L85) — 单函数、单次遍历、无复杂逻辑：

```cpp
PreCaptureResult pre_capture(const GraphAtlas& compile_atlas, Device* device) {
    std::unordered_map<CapturedGraph::Key, int32_t, CapturedGraph::KeyHash> seen;

    for (size_t vi = 0; vi < 6; ++vi) {
        for (uint8_t gi = 0; gi < GraphId::COUNT; ++gi) {
            auto& slot = result.atlas.slot(vi, gi);
            if (!slot.cg || !slot.mp) continue;

            Key key{slot.cg, static_cast<GraphId>(gi), slot.shape_id};

            auto it = seen.find(key);
            if (it != seen.end()) {
                slot.captured_idx = it->second;     // 命中：直接复用
                ++result.reused;
            } else {
                auto cg = CapturedGraph::capture(*slot.cg, *slot.mp,
                                                  key.gid, slot.shape_id, device);
                slot.captured_idx = result.graphs.size();
                seen[key] = slot.captured_idx;
                result.graphs.push_back(std::move(cg));
                ++result.captured;
            }
        }
    }
    return result;
}
```

**复杂度**：O(102) 遍历 + O(1) 哈希查重。没有预排序、没有分组、没有分支瀑布。

**唯一的状态**：`seen` map。遍历顺序先 shape 无关图再 shape 相关图，或先训练再推理——不影响结果，因为 Key 的三元组唯一性不依赖遍历顺序。

## 五、GraphAtlas 三阶段填表

| 阶段 | 谁填 | 填什么 | 时机 |
|------|------|--------|------|
| **Phase A** | Compiler | `Slot{cg, mp, shape_id}` | 编译完成后 |
| **Phase B** | `pre_capture()` | `Slot::captured_idx` | 预热捕获期 |
| **Phase C** | -（只读） | 双向 array 直接索引 | 运行时 |

`Slot` 结构（[graph_atlas.h:L47](file:///r:\renaissance\include\renaissance\graph\graph_atlas.h#L47)）：

```cpp
struct Slot {
    const ComputationGraph* cg = nullptr;     // Phase A 填入
    const MemoryPlan*       mp = nullptr;     // Phase A 填入
    ShapeId                 shape_id{};       // Phase A 填入（或 kShapeInvariant）
    int32_t                 captured_idx = -1; // Phase B 填入（-1 = 未捕获）
};
```

102 个 Slot 以二维数组 `table_[6][17]` 存储。Phase C 运行时通过 `atlas_.index(variant, gid)` 一次数组读取出 `captured_idx`，再通过该索引访问 `graphs_[idx].launch(stream)`。

**交叉槽位为空**：train 变体不填推理图，val 变体不填训练图——对应 Slot 的 `cg == nullptr`，`pre_capture()` 直接 `continue` 跳过。

## 六、数量精算

|   | 种类 | 槽位 | 捕获 | 复用 |
|---|------|------|------|------|
| shape 无关 | 8 | 48 | 8 | 40 |
| 训练 shape 相关 | ~4~8 | 16~32 | 16~32 | 0 |
| 推理 shape 相关 | ~2~4 | 4~8 | 4~8 | 0 |
| **合计** |   | **68~88** | **28~48** | **40** |

shape 无关图 100% 复用；shape 相关图按需捕获（不同 shape 各一张，相同 shape 共享一张）。

## 七、运行时 O(1)

去重仅在 Phase B 发生一次（MLPerf 不计时）。Phase C 运行时零开销：

```cpp
int32_t idx = atlas_.index(variant, gid);   // table_[6][17] 数组直接读
graphs_[idx].launch(stream);                 // 纯指针调用，零 hash，零查找
```

去重的"重"指的是捕获开销，不是查表开销。查表开销从设计之初就是 O(1) 的——去重只是让 102 个槽位不会产生 102 次捕获罢了。