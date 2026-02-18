# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**renAIssance** (formerly tech-renaissance) is a lightweight, high-performance, multi-backend, refactored, cross-platform deep learning framework developed by the "Tech Renaissance Team".

Current version: **V3.14.1** (see `version.txt` - never modify this file)

### Core Design Philosophy

- **Static Graph**: All operations are explicitly defined through method calls (no dynamic computation graph)
- **Backend Decoupling**: Clean separation between Device → Tensor → Storage
- **Zero-Copy Philosophy**: All operations use `into()` methods to avoid memory allocations
- **Developer-Centric**: Optimized for development efficiency over user-friendliness

## Architecture Overview

### Six Major Modules

```
include/renaissance/
├── data/          # Data loading and preprocessing
├── device/        # Hardware abstraction (CPU, CUDA, MUSA)
├── model/         # Neural network layers (Conv, Linear, etc.)
├── trainer/       # Training workflow management
├── base/          # Core utilities (Logger, Exception, Registry)
└── utils/         # Helper tools (Profiler, Python integration)
```

### Key Component Interactions

```
GlobalRegistry (config)
       ↓
DataLoader (zero-copy data access)
       ↓
Preprocessor (M workers, static sample assignment)
       ↓
Device (CPU/CUDA/MUSA operations)
       ↓
Tensor/Storage (memory management)
```

**Critical Design Pattern**: The Preprocessor uses a static worker assignment scheme where worker `i`'s `k`-th call reads sample `(i + k×M)`. This ensures reproducible randomness and even distribution across workers.

## Build System

### Quick Start

**Windows** (PowerShell or cmd only, not Git Bash):
```cmd
# Configure environment (first time only)
python configure.py

# Build Release version
build.bat

# Or manually with VS Developer Command Prompt
cmake -G Ninja -S . -B build/windows-msvc-release -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CXX_COMPILER=cl -DCMAKE_TOOLCHAIN_FILE=T:/Softwares/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build/windows-msvc-release --parallel 30
```

**Linux**:
```bash
# Configure environment (first time only)
python3 configure.py

# Build Release version
./build.sh

# Or manually
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --parallel $(nproc)
```

### Build Modes

- **Release**: `-O3 -march=native` (production, logging disabled except WARN/ERROR)
- **Debug**: `-g -O0` (development, all logging enabled)

### Dependencies

Managed via **vcpkg** with scene-based auto-configuration:
- `oneDNN` (x86 CPU), `XNNPACK` (ARM/RISC-V), `Simd` (image preprocessing)
- `CUDA` (>=13.0), `cuDNN` (>=9.19.0), `NCCL` (>=2.28.0)
- `mimalloc` (memory pool), `libjpeg-turbo`, `zlib`, `libcurl`, `stb`

## Testing

### Test Organization

```
tests/
├── unit_tests/         # Component-level tests
│   ├── base/          # Exception, logging, registry
│   ├── data/          # DataLoader, Preprocessor
│   ├── device/        # CPU/CUDA/MUSA operations
│   └── utils/         # Profiler, Python session
├── integration_tests/ # End-to-end workflows
│   └── test_epoch_crc.cpp  # CRC verification across epochs
└── dependency/        # Third-party library tests
```

### Running Tests

**Run individual test** (Linux example):
```bash
./build/bin/tests/data/test_dataloader_basic
```

**Run with Debug logging**:
```bash
# Build Debug version first
mkdir -p build/linux-debug
cd build/linux-debug
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug ../..
cmake --build . --target test_dataloader_basic
# Run with full logging
./bin/tests/data/test_dataloader_basic
```

### Key Test Patterns

- **Exception Testing**: Use try-catch to verify `TR_CHECK` throws expected exception types
- **CRC Verification**: Data integrity tests use `calc_crc=true` in Preprocessor Config
- **Reproducibility Tests**: Verify identical outputs across multiple epochs with same seed

## Code Style and Conventions

### File Structure Requirements

- **Header files**: `include/renaissance/{module}/{class_name}.h`
- **Source files**: `src/{module}/{class_name}.cpp`
- **Main header**: `include/renaissance.h` (includes everything, only this in tests)
- **All headers**: Use `#pragma once` (no include guards)

### Naming and Style

- **Namespace**: All code in `namespace tr`
- **C++ Standard**: C++17 only (no C++20 features)
- **Style Guide**: Google C++ Style Guide
- **Comments**: **Chinese** only, no emoji, Doxygen format for classes/functions
- **Outputs**: **English** only (logs, error messages, exceptions)

### File Header Format

```cpp
/**
 * @file filename.h
 * @brief 模块中文名称
 * @version X.XX.XX
 * @date YYYY-MM-DD
 * @author 技术觉醒团队
 * @note 依赖项: dep1, dep2
 * @note 所属系列: module
 */
```

## Exception Handling and Logging

### Core Principle

**Logger** = Record normal flow (INFO/WARN) and caught exceptions (ERROR)
**TRException** = Interrupt execution with context chains

### Three Key Macros

```cpp
// 1. TR_CHECK - Parameter validation (90% of use cases)
TR_CHECK(size > 0, ValueError, "batch_size must be positive, got " << size);

// 2. TR_XXX_ERROR - Direct throw (conditional logic)
TR_MEMORY_ERROR("CUDA allocation failed: " << size << " bytes");

// 3. TR_RETHROW - Add context in catch block
try {
    load_weights(path);
} catch (TRException& e) {
    TR_RETHROW(e, "Failed to load model '" << name_ << "' from '" << path << "'");
}
```

### Nine Exception Types

`NotImplementedError`, `ValueError`, `ShapeError`, `TypeError`, `IndexError`, `DeviceError`, `FileNotFoundError`, `ZeroDivisionError`, `MemoryError`

### Critical Rules

- ❌ **NEVER** use `LOG_ERROR + throw` (terminate handler auto-outputs)
- ✅ Use `TR_CHECK` or `TR_XXX_ERROR` directly
- ✅ For caught exceptions: `Logger::instance().log_exception(e)` or `LOG_ERROR << e.what()`
- ✅ Stream syntax: `<<` for variable interpolation (no `+` or `std::to_string()`)

### Logging

```cpp
LOG_INFO << "Epoch " << epoch << " started";
LOG_WARN << "GPU memory usage " << usage << "%";
LOG_DEBUG << "Tensor shape: [N,C,H,W]";

// Module-specific logging
TR_LOG_INFO("model") << "Building ResNet-50";
TR_LOG_DEBUG("data") << "Loading MNIST dataset";
```

## Global Registry Pattern

The `GlobalRegistry` singleton manages shared configuration with two types of variables:

### Fixed Variables (one-time assignment)

- `dataset_type`, `num_load_workers`, `num_preproc_workers`
- `world_size`, `batch_size`, `max_resolution`
- `num_color_channels`, `sdmp_factor`, `using_cpvs`

### Alterable Variables (modifiable when not busy)

- `current_resolution` (only when `is_busy() == false`)

### Phase Management

```cpp
GlobalRegistry::instance().begin_train();  // train_counter_++, is_busy may become true
// ... training code ...
GlobalRegistry::instance().end_train();    // train_counter_--
```

**Critical**: `initialize()` is automatically called on first `begin_train()`/`begin_val()`, validates all fixed variables are assigned.

## Key Design Decisions

### NHWC Data Layout

All tensors use channels-last format (N-H-W-C) for optimal image processing performance and hardware compatibility.

### Backend Decoupling

```
Device (abstract base)
  ├── CpuDevice (oneDNN/XNNPACK)
  ├── CudaDevice (cuDNN/cuBLAS)
  └── MusaDevice (muDNN)
```

Tensor operations are performed by calling device methods explicitly, not via tensor methods.

### Zero-Copy into() Methods

All operations follow this pattern to avoid allocations:

```cpp
// DON'T: result = device.add(a, b);  // Allocates new tensor
// DO: device.add_into(a, b, result);  // Writes to existing tensor
```

This is critical for performance - achieved ~50% speedup in tech-renaissance V2.

### Memory Architecture

- **CPU**: mimalloc pool (256-byte alignment)
- **GPU**: cudaMallocAsync for async memory management
- **Storage**: Two modes - Ownership (RAII) or Borrowing (MemoryArena-backed)

### Data Loading Pipeline

**Preprocessor V4.0** uses persistent worker threads with static sample assignment:
- Workers created once, process multiple buffers until epoch end
- Worker `i`'s `k`-th call → sample `(i + k×M)`
- Supports PARTIAL mode (multi-buffer loading) and FULLY mode (one-time load)

## Common Development Tasks

### Adding a New Layer

1. Create `include/renaissance/model/{layer_name}.h`
2. Inherit from `Module` base class
3. Implement `forward()` with proper shape validation using `TR_CHECK`
4. Add CMakeLists.txt entry in `src/model/`
5. Create unit test in `tests/unit_tests/model/`

### Adding CUDA Kernel

1. Create `src/device/cuda/{operation}_impl.cu`
2. Implement kernel with `extern "C"` linkage
3. Declare in `include/renaissance/device/cuda_device.h`
4. Link in `src/device/CMakeLists.txt`
5. Test in `tests/unit_tests/device/`

### Configuring for New Hardware

Edit `config/cmake_paths.cmake` or run `python configure.py` to regenerate based on detected hardware.

## Important Constraints

- **C++20 features**: Strictly prohibited (use C++17 only)
- **Emoji**: Forbidden in any code, comments, or outputs
- **Chinese in outputs**: Forbidden (use English only)
- **Direct TensorFlow/PyTorch dependency**: Prohibited (this is a standalone framework)
- **Version.txt**: Never modify `version.txt` manually

## Workspace and Temporary Files

All temporary files go to `workspace/` directory at project root (auto-created). Use `TR_WORKSPACE` macro to reference in code.

## Debugging

### Enable Full Logging

```cpp
Logger::instance().set_level(LogLevel::DEBUG);
```

### Check Data Integrity

```cpp
Preprocessor::Config config;
config.calc_crc = true;  // Enable CRC32 calculation
config.enable_logging = true;  // Output CSV logs
prep.configure(config);
```

### Verify Random Reproducibility

Use `tests/integration/test_epoch_crc.cpp` to verify identical CRC values across multiple epochs with same seed.

## References

- Design docs: `renaissance_prompt.md`, `renaissance_prompt_2.md`
- Project rules: `docs/rules.md`
- Exception handbook: `docs/logger_exception_handbook.md`
- Build guide: `docs/alpha_build.md`
