# 第 7 讲：Building a GPT

[仓库首页](../../../README.md) / [Zero to Hero](../README.md) / Building a GPT

从字符级 Bigram 到 GPT，学习因果自注意力、多头注意力、前馈网络、残差连接和 LayerNorm。

## 内容

- [gpt_from_scratch.ipynb](notebooks/gpt_from_scratch.ipynb)：课堂笔记与分步骤练习。
- [gpt.py](gpt.py)：完整模型、训练和文本生成代码。
- [requirements.txt](requirements.txt)：运行依赖。

## 运行

在本章节目录下安装依赖：

```powershell
python -m pip install -r requirements.txt
```

课堂笔记：打开 `notebooks/gpt_from_scratch.ipynb`，选择 Python 环境，按各节练习逐段运行。

完整代码：先下载 Tiny Shakespeare 数据到本章节目录，再运行完整脚本：

```powershell
python -c "from urllib.request import urlretrieve; urlretrieve('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', 'input.txt')"
python gpt.py
```

## 参考资料

- [B 站视频 · P7](https://www.bilibili.com/video/BV1mqrTBvEaf/?p=7)
- [本讲原始代码](https://github.com/karpathy/ng-video-lecture)
- [Tiny Shakespeare 数据](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare)
