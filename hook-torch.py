# hook-torch.py
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

# 收集 torch 的所有动态库
binaries = collect_dynamic_libs('torch')

# 收集 torch 的所有子模块
hiddenimports = collect_submodules('torch')