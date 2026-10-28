# 第 7 讲：Building a GPT

[仓库首页](../../../README.md) / [Zero to Hero](../README.md) / Building a GPT

从字符级 Bigram 模型出发，逐步加入因果自注意力、多头注意力、前馈网络、残差连接和 LayerNorm，构建一个可以训练和生成文本的小型 GPT。

## 代码

- [gpt.ipynb](notebooks/gpt.ipynb)：分步骤跟写、训练和生成文本。
- [requirements.txt](requirements.txt)：运行依赖。
- [download_data.py](scripts/download_data.py)：下载 Tiny Shakespeare 数据集。

## 运行

在本章节目录下安装依赖并下载数据：

```powershell
python -m pip install -r requirements.txt
python scripts/download_data.py
```

打开 [gpt.ipynb](notebooks/gpt.ipynb)，选择 Python 环境并按顺序运行。Notebook 的工作目录设为 `notebooks/`，数据文件为 `notebooks/input.txt`。

## 参考资料

- [B 站视频 · P7](https://www.bilibili.com/video/BV1mqrTBvEaf/?p=7)
- [本讲原始代码](https://github.com/karpathy/ng-video-lecture)
- [Tiny Shakespeare 数据](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare)
