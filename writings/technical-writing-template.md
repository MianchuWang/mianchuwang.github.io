---
title: "Technical Writing Template: Everything This Site Can Render"
date: 2026-07-21
tags: [meta]
draft: true
summary: A reference post for technical writing — math, code, tables, callouts, and every other supported Markdown feature in one place.
---

To publish a new post, drop a `.md` file into `content/` and push. The frontmatter declares the metadata:

```markdown
---
title: Post Title
date: 2026-07-21
tags: [tag-one, tag-two]
summary: One-line summary used as the page description.
---
```

Add `lang: zh` for posts written in Chinese, and `draft: true` to keep a post off the site. Below is a tour of everything the renderer supports.

## Mathematics

Inline math uses single dollar signs: let the loss be $\mathcal{L}(\theta) = \mathbb{E}_{x \sim p}[\ell(x; \theta)]$ with gradient $\nabla_\theta \mathcal{L}$.

Display math uses double dollar signs:

$$
\int_{-\infty}^{\infty} e^{-x^2}\, dx = \sqrt{\pi}
$$

Multi-line derivations use the `aligned` environment:

$$
\begin{aligned}
D_{\mathrm{KL}}(p \,\|\, q) &= \int p(x) \log \frac{p(x)}{q(x)}\, dx \\
&= \mathbb{E}_{x \sim p}\big[\log p(x) - \log q(x)\big] \\
&\geq 0
\end{aligned}
$$

Matrices work too:

$$
\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
\begin{pmatrix} x \\ y \end{pmatrix}
$$

## Code

Inline code: `torch.einsum("bqd,bkd->bqk", q, k)`. Code blocks get syntax highlighting and a hover copy button:

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

## Callouts

A blockquote starting with `[!note]`, `[!warn]`, or `[!important]` renders as a Notion-style callout:

> [!note]
> This is a note. Good for intuition, side remarks, or "why not do it the other way".

> [!warn]
> This is a warning. Good for pitfalls.

A plain blockquote stays a blockquote:

> The best way to predict the future is to invent it.

## Tables and lists

| Method | Sampling steps | Retraining |
| --- | ---: | :---: |
| DDPM | 1000 | — |
| DDIM | 20–50 | no |
| Distillation | 1–4 | yes |

- Unordered list
  - with nesting
- Works as expected

1. Step one
2. Step two

## Everything else

Horizontal rules, **bold**, *italic*, ~~strikethrough~~, [links](https://github.com/MianchuWang), and superscripts like H<sup>2</sup>O all work.

---

Long posts get an automatic table of contents on wide screens, and every heading has a `#` anchor link on hover.
