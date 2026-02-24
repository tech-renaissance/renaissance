# FRRC新旧版对比分析

## 问题1: SDMP=1时行为是否与RandomResizedCrop完全相同？

### 1.1 get_decode_strategy()对比

#### 旧版 (fast_random_resized_crop_old.cpp:58-97)
```cpp
DecodeStrategy get_decode_strategy(..., int sdmp_factor, ...) {
    sdmp_factor_ = sdmp_factor;
    if (sdmp_factor == 1) {
        crop_power_ = 3.0f;
    }
    // ...
    strategy.use_partial = true;
    auto* self = const_cast<FastRandomResizedCrop*>(this);
    self->generate_crop_params_for_partial(...);  // 总是用partial
    // ...
}
```

#### 新版 (src/data/fast_random_resized_crop.cpp:58-101)
```cpp
DecodeStrategy get_decode_strategy(..., int sdmp_factor, ...) {
    sdmp_factor_ = sdmp_factor;
    if (sdmp_factor == 1) {
        crop_power_ = 3.0f;
    }
    // ...
    strategy.use_partial = true;
    auto* self = const_cast<FastRandomResizedCrop*>(this);
    if (sdmp_factor == 1) {
        self->generate_crop_params_for_full(...);  // SDMP=1用full
    } else {
        self->generate_crop_params_for_partial(...);  // SDMP>1用partial
    }
    // ...
}
```

**关键差异**：
- ✅ 新版正确地在SDMP=1时调用了`generate_crop_params_for_full()`
- ❌ 旧版无论sdmp_factor是多少，都调用`generate_crop_params_for_partial()`

### 1.2 generate_crop_params_for_full() 对比 RandomResizedCrop::generate_crop_params()

#### 新版FRRC的generate_crop_params_for_full() (169-221行)
```cpp
void generate_crop_params_for_full(...) {
    const float area = static_cast<float>(image_width * image_height);
    const float log_scale_min = std::log(scale_min_);  // ← 对数空间
    const float log_scale_max = std::log(scale_max_);
    const float log_ratio_min = std::log(ratio_min_);
    const float log_ratio_max = std::log(ratio_max_);

    constexpr int MAX_ATTEMPTS = 10;
    bool success = false;

    for (int attempt = 0; attempt < MAX_ATTEMPTS && !success; ++attempt) {
        uint64_t scale_offset = rng->next_offset(1);
        float scale_rand = detail::philox_uniform_float(rng->seed(), scale_offset);
        const float target_area = area * std::exp(  // ← 对数空间采样
            log_scale_min + scale_rand * (log_scale_max - log_scale_min));

        // ... 完全相同的逻辑 ...
    }
    // ... 完全相同的fallback逻辑 ...
}
```

#### RandomResizedCrop的generate_crop_params() (random_resized_crop.cpp:103-185)
```cpp
void generate_crop_params(...) {
    const float area = static_cast<float>(image_width * image_height);
    const float log_scale_min = std::log(scale_min_);  // ← 对数空间
    const float log_scale_max = std::log(scale_max_);
    const float log_ratio_min = std::log(ratio_min_);
    const float log_ratio_max = std::log(ratio_max_);

    constexpr int MAX_ATTEMPTS = 10;
    bool success = false;

    for (int attempt = 0; attempt < MAX_ATTEMPTS && !success; ++attempt) {
        uint64_t scale_offset = rng->next_offset(1);
        float scale_rand = detail::philox_uniform_float(rng->seed(), scale_offset);
        const float target_area = area * std::exp(  // ← 对数空间采样
            log_scale_min + scale_rand * (log_scale_max - log_scale_min));

        // ... 完全相同的逻辑 ...
    }
    // ... 完全相同的fallback逻辑 ...
}
```

**验证结果**: ✅ **完全一致！两个函数逐行相同**

### 1.3 execute_from_partial_decode() 对比

#### 新版FRRC (317-376行)
```cpp
void execute_from_partial_decode(...) {
    if (sdmp_factor_ == 1) {  // ← 关键分支
        // 行为与RandomResizedCrop一致
        int offset_x = crop_x_ - mcu_x_;
        int offset_y = crop_y_ - mcu_y_;
        const uint8_t* src_row = input_ptr + offset_y * input_stride + offset_x * num_channels_;
        void* resizer = SimdResizerInit(crop_w_, crop_h_, output_size_, output_size_, ...);
        SimdResizerRun(resizer, src_row, input_stride, output_ptr, output_stride);
        SimdRelease(resizer);
    } else {
        // FRRC的两阶段随机
        // ... 第二次随机crop ...
    }
}
```

#### RandomResizedCrop (random_resized_crop.cpp:320-348)
```cpp
void execute_from_partial_decode(...) {
    // 总是单次crop
    int offset_x = crop_x_ - mcu_x_;
    int offset_y = crop_y_ - mcu_y_;
    const uint8_t* src_row = input_ptr + offset_y * input_stride + offset_x * num_channels_;
    void* resizer = SimdResizerInit(crop_w_, crop_h_, output_size_, output_size_, ...);
    SimdResizerRun(resizer, src_row, input_stride, output_ptr, output_stride);
    SimdRelease(resizer);
}
```

**验证结果**: ✅ **当sdmp_factor_==1时，行为与RandomResizedCrop完全一致**

---

## 问题2: SDMP>1时行为是否与旧版FRRC完全相同？

### 2.1 get_decode_strategy()对比

#### 旧版 (fast_random_resized_crop_old.cpp:58-97)
```cpp
strategy.use_partial = true;
self->generate_crop_params_for_partial(image_width, image_height, rng);  // 总是partial
self->calculate_mcu_aligned_region(image_width, image_height);
```

#### 新版 (src/data/fast_random_resized_crop.cpp:58-101)
```cpp
strategy.use_partial = true;
if (sdmp_factor == 1) {
    self->generate_crop_params_for_full(image_width, image_height, rng);
} else {
    self->generate_crop_params_for_partial(image_width, image_height, rng);  // SDMP>1用partial
}
self->calculate_mcu_aligned_region(image_width, image_height);
```

**验证结果**: ✅ **当sdmp_factor>1时，都调用generate_crop_params_for_partial()**

### 2.2 generate_crop_params_for_partial()对比

#### 旧版 (fast_random_resized_crop_old.cpp:99-163)
```cpp
void generate_crop_params_for_partial(...) {
    const float area = static_cast<float>(image_width * image_height);
    const float log_ratio_min = std::log(ratio_min_);
    const float log_ratio_max = std::log(ratio_max_);

    constexpr int MAX_ATTEMPTS = 10;
    bool success = false;

    for (int attempt = 0; attempt < MAX_ATTEMPTS && !success; ++attempt) {
        uint64_t scale_offset = rng->next_offset(1);
        float scale_rand = detail::philox_uniform_float(rng->seed(), scale_offset);
        const float target_area = area * std::pow(
            sqrt3_scale_min_ + scale_rand * (sqrt3_scale_max_ - sqrt3_scale_min_),
            crop_power_);  // ← crop_power_由sdmp_factor决定

        // ... 其余逻辑完全相同 ...
    }
    // ... 完全相同的fallback逻辑 ...
}
```

#### 新版 (src/data/fast_random_resized_crop.cpp:103-167)
```cpp
void generate_crop_params_for_partial(...) {
    const float area = static_cast<float>(image_width * image_height);
    const float log_ratio_min = std::log(ratio_min_);
    const float log_ratio_max = std::log(ratio_max_);

    constexpr int MAX_ATTEMPTS = 10;
    bool success = false;

    for (int attempt = 0; attempt < MAX_ATTEMPTS && !success; ++attempt) {
        uint64_t scale_offset = rng->next_offset(1);
        float scale_rand = detail::philox_uniform_float(rng->seed(), scale_offset);
        const float target_area = area * std::pow(
            sqrt3_scale_min_ + scale_rand * (sqrt3_scale_max_ - sqrt3_scale_min_),
            crop_power_);  // ← crop_power_由sdmp_factor决定

        // ... 其余逻辑完全相同 ...
    }
    // ... 完全相同的fallback逻辑 ...
}
```

**验证结果**: ✅ **完全相同！**

### 2.3 crop_power_设置对比

#### 旧版 (fast_random_resized_crop_old.cpp:64-80)
```cpp
sdmp_factor_ = sdmp_factor;
if (sdmp_factor == 1) {
    crop_power_ = 3.0f;
}
else if (sdmp_factor == 2) {
    crop_power_ = 2.0f;
}
else if (sdmp_factor == 3) {
#ifdef CROP_SCHEME_2
    crop_power_ = 2.0f;
#else
    crop_power_ = 1.0f;
#endif
}
else {
    crop_power_ = 1.0f;
}
```

#### 新版 (src/data/fast_random_resized_crop.cpp:64-80)
```cpp
sdmp_factor_ = sdmp_factor;
if (sdmp_factor == 1) {
    crop_power_ = 3.0f;
}
else if (sdmp_factor == 2) {
    crop_power_ = 2.0f;
}
else if (sdmp_factor == 3) {
#ifdef CROP_SCHEME_2
    crop_power_ = 2.0f;
#else
    crop_power_ = 1.0f;
#endif
}
else {
    crop_power_ = 1.0f;
}
```

**验证结果**: ✅ **完全相同！**

### 2.4 execute_from_partial_decode()的两阶段随机对比

#### 旧版 (fast_random_resized_crop_old.cpp:344-382)
```cpp
else {  // sdmp_factor_ != 1
    uint64_t scale_offset = rng->next_offset(1);
    float scale_rand = detail::philox_uniform_float(rng->seed(), scale_offset);
    const float target_area = crop_w_ * crop_h_ * std::pow(
        sqrt3_scale_min_ + scale_rand * (sqrt3_scale_max_ - sqrt3_scale_min_),
        (3.0f - crop_power_));

    int crop_w = static_cast<int>(std::floor(std::sqrt(target_area * aspect_ratio_)));
    int crop_h = static_cast<int>(std::floor(std::sqrt(target_area / aspect_ratio_)));

    int max_offset_x = mcu_w_ - crop_w + 1;
    int max_offset_y = mcu_h_ - crop_h + 1;

    uint64_t x_offset = rng->next_offset(1);
    float x_rand = detail::philox_uniform_float(rng->seed(), x_offset);
    int crop_x_t = static_cast<int>(x_rand * max_offset_x);

    uint64_t y_offset = rng->next_offset(1);
    float y_rand = detail::philox_uniform_float(rng->seed(), y_offset);
    int crop_y_t = static_cast<int>(y_rand * max_offset_y);

    crop_x_t = std::min(crop_x_t, mcu_w_ - crop_w);
    crop_y_t = std::min(crop_y_t, mcu_h_ - crop_h);

    int offset_x = crop_x_t;
    int offset_y = crop_y_t;

    const uint8_t* src_row = input_ptr + offset_y * input_stride + offset_x * num_channels_;
    void* resizer = SimdResizerInit(crop_w, crop_h, output_size_, output_size_, ...);
    SimdResizerRun(resizer, src_row, input_stride, output_ptr, output_stride);
    SimdRelease(resizer);
}
```

#### 新版 (src/data/fast_random_resized_crop.cpp:337-375)
```cpp
else {  // sdmp_factor_ != 1
    uint64_t scale_offset = rng->next_offset(1);
    float scale_rand = detail::philox_uniform_float(rng->seed(), scale_offset);
    const float target_area = crop_w_ * crop_h_ * std::pow(
        sqrt3_scale_min_ + scale_rand * (sqrt3_scale_max_ - sqrt3_scale_min_),
        (3.0f - crop_power_));

    int crop_w = static_cast<int>(std::floor(std::sqrt(target_area * aspect_ratio_)));
    int crop_h = static_cast<int>(std::floor(std::sqrt(target_area / aspect_ratio_)));

    int max_offset_x = mcu_w_ - crop_w + 1;
    int max_offset_y = mcu_h_ - crop_h + 1;

    uint64_t x_offset = rng->next_offset(1);
    float x_rand = detail::philox_uniform_float(rng->seed(), x_offset);
    int crop_x_t = static_cast<int>(x_rand * max_offset_x);

    uint64_t y_offset = rng->next_offset(1);
    float y_rand = detail::philox_uniform_float(rng->seed(), y_offset);
    int crop_y_t = static_cast<int>(y_rand * max_offset_y);

    crop_x_t = std::min(crop_x_t, mcu_w_ - crop_w);
    crop_y_t = std::min(crop_y_t, mcu_h_ - crop_h);

    int offset_x = crop_x_t;
    int offset_y = crop_y_t;

    const uint8_t* src_row = input_ptr + offset_y * input_stride + offset_x * num_channels_;
    void* resizer = SimdResizerInit(crop_w, crop_h, output_size_, output_size_, ...);
    SimdResizerRun(resizer, src_row, input_stride, output_ptr, output_stride);
    SimdRelease(resizer);
}
```

**验证结果**: ✅ **完全相同！**

---

## 问题3: 局部解码失败时行为是否与RandomResizedCrop完全相同？

### execute()方法的逻辑

#### 新版FRRC (249-282行)
```cpp
void execute(..., bool execute_from_full, ...) {
    if (execute_from_full || !rank_first_in_the_po_chain_) {
        generate_crop_params_for_full(input_width, input_height, rng);  // ← RRC策略
    }

    // ...

    if (execute_from_full || !rank_first_in_the_po_chain_) {
        execute_from_full_decode(...);  // ← RRC行为
    } else {
        execute_from_partial_decode(..., rng);
    }
}
```

**关键点**：
- `execute_from_full=true` 表示局部解码失败或不用解码
- 此时调用 `generate_crop_params_for_full()` → **RRC策略**
- 此时调用 `execute_from_full_decode()` → **RRC行为**

**验证结果**: ✅ **局部解码失败时，行为与RandomResizedCrop完全一致**

---

## 总结

### ✅ 验证通过

1. **SDMP=1时**：
   - get_decode_strategy()调用generate_crop_params_for_full()
   - generate_crop_params_for_full()与RRC.generate_crop_params()**逐行相同**
   - execute_from_partial_decode()中sdmp_factor_==1分支与RRC**完全相同**
   - **结论：行为与RandomResizedCrop完全相同**

2. **SDMP>1时**：
   - get_decode_strategy()调用generate_crop_params_for_partial()
   - generate_crop_params_for_partial()与旧版**完全相同**
   - crop_power_设置与旧版**完全相同**
   - execute_from_partial_decode()的两阶段随机与旧版**完全相同**
   - **结论：行为与旧版FRRC完全相同**

3. **局部解码失败时**：
   - execute()调用generate_crop_params_for_full() → **RRC策略**
   - execute()调用execute_from_full_decode() → **RRC行为**
   - **结论：行为与RandomResizedCrop完全相同**
