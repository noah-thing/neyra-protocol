"""Draw the README figures. The terminal text is real output, copied from a run, not composed.

social-preview.png and icon.png come from the site, so GitHub and neyra.run show the same mark.
"""

from html import escape
from pathlib import Path

HERE = Path(__file__).parent
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
THEMES = {
    "light": dict(bg="#ffffff", panel="#f6f8fa", line="#d0d7de", text="#1f2328", dim="#656d76", cell="#d8dee4", hit="#1f2328", bad="#cf222e"),
    "dark": dict(bg="#0d1117", panel="#161b22", line="#30363d", text="#e6edf3", dim="#8b949e", cell="#30363d", hit="#ffffff", bad="#ff7b72"),
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

VERDICT = [
    ("cmd", "$ neyra tune verdict Qwen/Qwen2.5-0.5B-Instruct ./adapter cases.jsonl --examples train.jsonl --shots 40 --gguf *.gguf"),
    ("", ""),
    ("hit", "This worked: the tuned model scores 98.5% on 200 cases, against 62.5% for the base and 90.0% with 40 examples in the"),
    ("hit", "prompt; it is +8.5 points ahead of the best free option, few-shot (interval +5.0 to +12.5)"),
    ("", ""),
    ("dim", "          score    against base                    against few-shot"),
    ("out", "base      62.5%"),
    ("out", "few-shot  90.0%     +27.5 [+21.0, +34.5] +60/-5"),
    ("out", "tuned     98.5%     +36.0 [+29.5, +42.5] +73/-1      +8.5 [+5.0, +12.5] +17/-0"),
    ("out", "F16       98.5%     +36.0 [+29.5, +42.5] +73/-1      +8.5 [+5.0, +12.5] +17/-0"),
    ("out", "Q8_0      98.5%     +36.0 [+29.5, +42.5] +73/-1      +8.5 [+5.0, +12.5] +17/-0"),
    ("out", "Q4_K_M    98.5%     +36.0 [+29.5, +42.5] +73/-1      +8.5 [+5.0, +12.5] +17/-0"),
    ("out", "Q3_K_S    97.5%     +35.0 [+28.0, +42.0] +72/-2      +7.5 [+4.0, +11.5] +16/-1"),
    ("out", "Q2_K      97.5%     +35.0 [+28.5, +41.5] +71/-1      +7.5 [+4.0, +11.5] +16/-1"),
    ("", ""),
    ("dim", "Damage"),
    ("out", "  general questions: base 83.3%, tuned 96.7% of 30 (+13.3 points, interval +3.3 to +26.7)"),
    ("out", "  cases the base got right and the tuned model got wrong: 1 of 200"),
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


def session(t, lines, label):
    w, line, pad = 1000, 22, 22
    h = pad * 2 + 30 + line * len(lines)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="{label}">',
           f'<rect x=".5" y=".5" width="{w - 1}" height="{h - 1}" rx="8" fill="{t["panel"]}" stroke="{t["line"]}"/>']
    for i in range(3):
        out.append(f'<circle cx="{pad + i * 18}" cy="20" r="5" fill="{t["line"]}"/>')
    colours = {"cmd": t["text"], "out": t["text"], "dim": t["dim"], "hit": t["hit"]}
    y = pad + 32
    for kind, text in lines:
        if text:
            weight = ' font-weight="600"' if kind == "cmd" else ""
            out.append(f'<text x="{pad}" y="{y}" xml:space="preserve" font-family="{MONO}" font-size="13"{weight} fill="{colours[kind]}">{escape(text)}</text>')
        y += line
    return "".join(out) + "</svg>"


def terminal(t, dark):
    return session(t, SESSION, "neyra diff recovering a published edit from a 4 percent read")


def verdict(t, dark):
    return session(t, VERDICT, "neyra tune verdict scoring a public LoRA against the base, a prompt and quantized files")


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


for name, draw in (("logo", logo), ("terminal", terminal), ("verdict", verdict), ("hops", hops)):
    for theme, colours in THEMES.items():
        (HERE / f"{name}-{theme}.svg").write_text(draw(colours, theme == "dark"))
