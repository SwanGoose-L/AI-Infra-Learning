from pathlib import Path
from urllib.request import urlretrieve

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
target = Path(__file__).resolve().parents[1] / "notebooks" / "input.txt"
if target.exists():
    print(f"已存在，保留原文件：{target}")
else:
    urlretrieve(url, target)
    print(f"下载完成：{target}")
