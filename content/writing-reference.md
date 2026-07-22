---
title: 排版参考：这个站点支持的全部写法
date: 2026-07-21
tags: [meta]
summary: 一篇自用的样式参考——数学公式、代码、表格、提示框等所有支持的 Markdown 语法都在这里过一遍。
---

新文章只需要在 `content/` 里新建一个 `.md` 文件，push 之后自动上线。文件开头用 frontmatter 声明元信息：

```markdown
---
title: 文章标题
date: 2026-07-21
tags: [tag-one, tag-two]
summary: 显示在首页列表里的一句话摘要。
---
```

下面把支持的语法过一遍。

## 数学公式

行内公式用单个美元符号：设损失函数为 $\mathcal{L}(\theta) = \mathbb{E}_{x \sim p}[\ell(x; \theta)]$，梯度为 $\nabla_\theta \mathcal{L}$。

独立公式用两个美元符号：

$$
\int_{-\infty}^{\infty} e^{-x^2}\, dx = \sqrt{\pi}
$$

多行对齐用 `aligned` 环境：

$$
\begin{aligned}
D_{\mathrm{KL}}(p \,\|\, q) &= \int p(x) \log \frac{p(x)}{q(x)}\, dx \\
&= \mathbb{E}_{x \sim p}\big[\log p(x) - \log q(x)\big] \\
&\geq 0
\end{aligned}
$$

矩阵也没问题：

$$
\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
\begin{pmatrix} x \\ y \end{pmatrix}
$$

## 代码

行内代码：`torch.einsum("bqd,bkd->bqk", q, k)`。代码块带语法高亮和悬停复制按钮：

```python
@torch.no_grad()
def sample(model, shape, T, alpha, alpha_bar, sigma):
    x = torch.randn(shape)
    for t in reversed(range(1, T + 1)):
        eps = model(x, t)
        x = (x - (1 - alpha[t]) / (1 - alpha_bar[t]).sqrt() * eps) / alpha[t].sqrt()
        if t > 1:
            x += sigma[t] * torch.randn_like(x)
    return x
```

```bash
python3 scripts/build_manifest.py && python3 -m http.server 8000
```

## 提示框

引用块首行写 `[!note]`、`[!warn]`、`[!important]` 会渲染成 Notion 风格的 callout：

> [!note]
> 这是一个提示框。适合放补充说明、直觉解释，或者"为什么不这样做"。

> [!warn]
> 这是一个警告框。适合放容易踩的坑。

普通引用还是普通引用：

> The best way to predict the future is to invent it.

## 表格与列表

| 方法 | 采样步数 | 是否需要重训练 |
| --- | ---: | :---: |
| DDPM | 1000 | — |
| DDIM | 20–50 | 否 |
| 蒸馏 | 1–4 | 是 |

- 无序列表
  - 支持嵌套
- 有序列表也一样

1. 第一步
2. 第二步

## 其他

分割线、**加粗**、*斜体*、~~删除线~~、[链接](https://github.com/MianchuWang)，以及脚注风格的上标 H<sup>2</sup>O 都正常工作。

---

长标题的文章会在宽屏右侧自动生成目录，点击标题左侧的 `#` 可以拿到锚点链接。
