#!/usr/bin/env python3
"""
TR4 Build Helper — 纯 Python 脚本，供无法使用 PowerShell/cmd 的 AI 使用。
功能：清理 / configure / 全量重建 / 增量编译指定目标。
"""
import argparse
import os
import shutil
import subprocess
import sys

PROJECT_ROOT = r"R:\renaissance"
VCVARS_BAT = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
CMAKE_TOOLCHAIN = r"T:/Softwares/vcpkg/scripts/buildsystems/vcpkg.cmake"


def _path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def get_vcvars_env():
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bat', delete=False, encoding='ascii') as f:
        f.write(f'@echo off\ncall "{VCVARS_BAT}"\nset\n')
        tmp_bat = f.name
    try:
        result = subprocess.run(
            ['cmd.exe', '/c', tmp_bat],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
        )
    finally:
        os.remove(tmp_bat)
    env = dict(os.environ)
    for line in result.stdout.splitlines():
        if '=' in line and not line.startswith('='):
            key, _, value = line.partition('=')
            env[key] = value
    if 'INCLUDE' not in env and 'LIB' not in env:
        print("[ERROR] vcvars64.bat did not set INCLUDE/LIB environment variables")
        sys.exit(1)
    return env


def do_clear():
    for name in ("build", "config", "build.bat"):
        path = _path(name)
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"[CLEAR] Deleted directory: {path}")
        elif os.path.isfile(path):
            os.remove(path)
            print(f"[CLEAR] Deleted file: {path}")
        else:
            print(f"[CLEAR] Not found (skipped): {path}")


def do_config():
    print("[CONFIG] Running configure.py ...")
    proc = subprocess.Popen(
        [sys.executable, "configure.py"],
        cwd=PROJECT_ROOT,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    stdout, _ = proc.communicate(input="Y\n1\n")
    print(stdout)
    if proc.returncode != 0:
        print("[ERROR] configure.py failed")
        sys.exit(1)
    print("[CONFIG] Done")


def do_cmake_configure(env):
    print("[CMAKE] Configuring (MSBuild) ...")
    cmd = [
        "cmake",
        "-G", "Visual Studio 17 2022", "-A", "x64",
        "-S", PROJECT_ROOT,
        "-B", BUILD_DIR,
        f"-DCMAKE_TOOLCHAIN_FILE={CMAKE_TOOLCHAIN}",
        "-DTR_SCENE_PC_CUDA=ON",
        "-DTR_USE_CUDA=ON",
        "-DTR_USE_MUSA=OFF",
        "-DTR_USE_EIGEN=ON",
        "-DTR_USE_XNNPACK=ON",
        "-DTR_USE_STB=ON",
        "-DTR_USE_LIBJPEG=ON",
        "-DTR_USE_ZLIB=ON",
        "-DTR_USE_LIBCURL=ON",
        "-DTR_USE_LIBARCHIVE=ON",
        "-DTR_USE_MIMALLOC=ON",
        "-DTR_USE_SIMD=ON",
        "-DCMAKE_MSVC_PARALLEL=30",
    ]
    result = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print("[ERROR] CMake configuration failed")
        sys.exit(1)
    print("[CMAKE] Done")


def do_build(env, target=None):
    cmd = ["cmake", "--build", BUILD_DIR, "--config", "Release", "--parallel", "30"]
    if target:
        cmd.extend(["--target", target])
        print(f"[BUILD] Incremental: {target}")
    else:
        print("[BUILD] Full build")
    result = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print("[ERROR] Build failed")
        sys.exit(1)
    print("[BUILD] Done")


def main():
    parser = argparse.ArgumentParser(description="TR4 Build Helper")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--config", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--target", type=str)
    args = parser.parse_args()

    if args.clear:
        do_clear()
    if args.config:
        do_config()
    if args.rebuild or args.target:
        env = get_vcvars_env()
        do_cmake_configure(env)
        do_build(env, args.target)
    if not any((args.clear, args.config, args.rebuild, args.target)):
        parser.print_help()


if __name__ == "__main__":
    main()
