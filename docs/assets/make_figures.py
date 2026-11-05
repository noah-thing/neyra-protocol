"""Draw the README figures. The terminal text is real output, copied from a run, not composed."""

from html import escape
from pathlib import Path

HERE = Path(__file__).parent
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
THEMES = {
    "light": dict(bg="#ffffff", panel="#f6f8fa", line="#d0d7de", text="#1f2328", dim="#656d76", cell="#d8dee4", hit="#0969da", bad="#cf222e"),
    "dark": dict(bg="#0d1117", panel="#161b22", line="#30363d", text="#e6edf3", dim="#8b949e", cell="#30363d", hit="#58a6ff", bad="#ff7b72"),
}

SESSION = [
    ("cmd", "$ neyra diff Qwen/Qwen3.5-0.8B RikiyaT/Qwen3.5-0.8B-weight-space-programming --sample 4"),
    ("", ""),
    ("out", "488 tensors paired, 2 changed, no difference found in the other 486 (sampled spans only, which is not proof they match)"),
    ("out", "130,188 entries differ, 1.774% of the changed tensors"),
    ("dim", "read 144.2 MB of 3,493.9 MB in 172s"),
    ("", ""),
    ("out", "model.language_model.layers.15.mlp.down_proj.weight"),
    ("hit", "    W' = W - 6.000 u(uᵀW) on 64 of 3584 columns"),
    ("out", "model.language_model.layers.19.mlp.down_proj.weight"),
    ("hit", "    W' = W - 6.000 u(uᵀW) on 1023 of 1024 rows, 64 of 3584 columns"),
]


def logo(t, dark):
    cells, size, gap = 5, 26, 6
    squares = []
    for r in range(cells):
        for c in range(cells):
            fill = t["hit"] if (r, c) == (1, 3) else t["cell"]
            squares.append(f'<rect x="{c * (size + gap)}" y="{r * (size + gap)}" width="{size}" height="{size}" rx="3" fill="{fill}"/>')
    word = f'<text x="186" y="112" font-family="{SANS}" font-size="118" font-weight="600" letter-spacing="-3" fill="{t["text"]}">neyra</text>'
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 160" role="img" aria-label="Neyra">{"".join(squares)}{word}</svg>'


def terminal(t, dark):
    w, line, pad = 1000, 22, 22
    h = pad * 2 + 30 + line * len(SESSION)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="neyra diff recovering a published edit from a 4 percent read">',
           f'<rect x=".5" y=".5" width="{w - 1}" height="{h - 1}" rx="8" fill="{t["panel"]}" stroke="{t["line"]}"/>']
    for i in range(3):
        out.append(f'<circle cx="{pad + i * 18}" cy="20" r="5" fill="{t["line"]}"/>')
    colours = {"cmd": t["text"], "out": t["text"], "dim": t["dim"], "hit": t["hit"]}
    y = pad + 32
    for kind, text in SESSION:
        if text:
            weight = ' font-weight="600"' if kind == "cmd" else ""
            out.append(f'<text x="{pad}" y="{y}" xml:space="preserve" font-family="{MONO}" font-size="13"{weight} fill="{colours[kind]}">{escape(text)}</text>')
        y += line
    return "".join(out) + "</svg>"


def hops(t, dark):
    names = [("base", "published weights"), ("adapter", "your fine-tune"), ("merged", "safetensors"), ("GGUF", "quantized")]
    w, h, box, gap, top = 1000, 150, 212, 48, 34
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="A fine-tune passing through merge, conversion and serving, with the first differing hop marked">']
    for i, (name, note) in enumerate(names):
        x = 3 + i * (box + gap)
        lost = name == "merged"
        stroke = t["bad"] if lost else t["line"]
        out.append(f'<rect x="{x}" y="{top}" width="{box}" height="72" rx="6" fill="{t["panel"]}" stroke="{stroke}" stroke-width="{2 if lost else 1}"/>')
        out.append(f'<text x="{x + box / 2}" y="{top + 32}" text-anchor="middle" font-family="{SANS}" font-size="17" font-weight="600" fill="{t["text"]}">{name}</text>')
        out.append(f'<text x="{x + box / 2}" y="{top + 54}" text-anchor="middle" font-family="{SANS}" font-size="13" fill="{t["dim"]}">{note}</text>')
        if i < len(names) - 1:
            a = x + box + 8
            out.append(f'<path d="M{a} {top + 36}h{gap - 16}m-6 -5l6 5l-6 5" fill="none" stroke="{t["dim"]}" stroke-width="1.5"/>')
        if lost:
            out.append(f'<text x="{x + box / 2}" y="{top + 96}" text-anchor="middle" font-family="{SANS}" font-size="13" fill="{t["bad"]}">first hop that differs</text>')
    return "".join(out) + "</svg>"


def card(t, dark):
    cells, size, gap = 5, 44, 10
    x0, y0 = 250, 215
    squares = []
    for r in range(cells):
        for c in range(cells):
            fill = t["hit"] if (r, c) == (1, 3) else t["cell"]
            squares.append(f'<rect x="{x0 + c * (size + gap)}" y="{y0 + r * (size + gap) - 60}" width="{size}" height="{size}" rx="5" fill="{fill}"/>')
    word = f'<text x="540" y="330" font-family="{SANS}" font-size="170" font-weight="600" letter-spacing="-4" fill="{t["text"]}">neyra</text>'
    line = f'<text x="640" y="500" text-anchor="middle" font-family="{SANS}" font-size="34" fill="{t["dim"]}">Edit local model weights with patches that revert bit for bit.</text>'
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 640"><rect width="1280" height="640" fill="{t["bg"]}"/>{"".join(squares)}{word}{line}</svg>'


for name, draw in (("logo", logo), ("terminal", terminal), ("hops", hops), ("card", card)):
    for theme, colours in THEMES.items():
        (HERE / f"{name}-{theme}.svg").write_text(draw(colours, theme == "dark"))
