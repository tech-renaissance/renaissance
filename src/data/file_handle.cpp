/**
 * @file file_handle.cpp
 * @brief 跨平台文件句柄 RAII 封装实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#include "renaissance/data/file_handle.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"

#include <cstring>  // For strerror (Linux)

namespace tr {

// =============================================================================
// Windows实现
// =============================================================================

#ifdef _WIN32

FileHandle::FileHandle(const std::string& path) {
    handle_ = CreateFileA(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
        NULL
    );

    if (handle_ == INVALID_HANDLE_VALUE) {
        TR_FILE_NOT_FOUND("Failed to open " << path
                         << "\n  Error code: " << GetLastError());
    }

    LOG_DEBUG << "Opened file (Windows): " << path;
}

FileHandle::~FileHandle() {
    if (handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(handle_);
    }
}

FileHandle::FileHandle(FileHandle&& other) noexcept
    : handle_(other.handle_) {
    other.handle_ = INVALID_HANDLE_VALUE;
}

FileHandle& FileHandle::operator=(FileHandle&& other) noexcept {
    if (this != &other) {
        if (handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
        }
        handle_ = other.handle_;
        other.handle_ = INVALID_HANDLE_VALUE;
    }
    return *this;
}

// =============================================================================
// Linux实现
// =============================================================================

#else

FileHandle::FileHandle(const std::string& path) : handle_(-1) {
    handle_ = open(path.c_str(), O_RDONLY | O_LARGEFILE);
    if (handle_ < 0) {
        TR_FILE_NOT_FOUND("Failed to open " << path
                         << "\n  Error: " << strerror(errno));
    }

    LOG_DEBUG << "Opened file (Linux): " << path;
}

FileHandle::~FileHandle() {
    if (handle_ >= 0) {
        close(handle_);
    }
}

FileHandle::FileHandle(FileHandle&& other) noexcept
    : handle_(other.handle_) {
    other.handle_ = -1;
}

FileHandle& FileHandle::operator=(FileHandle&& other) noexcept {
    if (this != &other) {
        if (handle_ >= 0) {
            close(handle_);
        }
        handle_ = other.handle_;
        other.handle_ = -1;
    }
    return *this;
}

#endif

} // namespace tr
