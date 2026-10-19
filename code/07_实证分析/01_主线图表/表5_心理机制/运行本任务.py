"""表5独立运行入口，参数沿用本目录的估计与绘图脚本。"""

import sys
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    public_dir = parent / "90_公共模块"
    if (public_dir / "任务入口.py").is_file():
        sys.path.insert(0, str(public_dir))
        break
else:
    raise FileNotFoundError("未找到公共任务入口。")
from 任务入口 import _load_module


if __name__ == "__main__":
    _load_module(Path(__file__).with_name("估计回归与绘图.py")).main()
