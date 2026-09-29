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

Colored inline code marks a category that the text defines: <span class="code-blue">`batch`, `advantages`</span>, <span class="code-green">`uid`</span>, <span class="code-amber">`temperature`</span>.

Inline code: `torch.einsum("bqd,bkd->bqk", q, k)`. Code blocks get syntax highlighting and a hover copy button. A fence can carry `lines=` to show source line numbers; a line that is only `...` marks an elision and gets no number:

```python lines=40-44,50-53
@torch.no_grad()
def sample(model, shape, T, alpha, alpha_bar, sigma):
    x = torch.randn(shape)
    for t in reversed(range(1, T + 1)):
        eps = model(x, t)
    ...
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

## Diagrams

A diagram is an inline SVG in a `diagram` block. Its classes take their colors from the theme:

<div class="diagram">
<svg viewBox="0 0 760 120" role="img" aria-label="A driver box with an arrow to a model box and a rollout box.">
<defs><marker id="dg-arrow-t" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="head" d="M0,0 L10,5 L0,10 z"/></marker></defs>
<rect class="proc" x="10" y="10" width="740" height="100" rx="8"/>
<rect class="box" x="30" y="30" width="180" height="60" rx="6"/><text class="m" x="44" y="56">driver</text><text class="s" x="44" y="74">class box</text>
<rect class="model" x="290" y="30" width="180" height="60" rx="6"/><text class="t" x="304" y="56">model engine</text><text class="s" x="304" y="74">class model</text>
<rect class="rollout" x="550" y="30" width="180" height="60" rx="6"/><text class="t" x="564" y="56">rollout engine</text><text class="s" x="564" y="74">class rollout</text>
<path class="arrow" d="M210,60 L289,60" marker-end="url(#dg-arrow-t)"/><path class="arrow" d="M470,60 L549,60" marker-end="url(#dg-arrow-t)"/>
</svg>
</div>

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
