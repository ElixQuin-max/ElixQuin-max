#!/usr/bin/env python3
"""Genera todo el perfil de GitHub a partir de profile.config.json.

Uso (desde la raíz del repo):
    pip install -r scripts/requirements.txt
    python scripts/generate.py

Produce:
    assets/banner-dark.svg / banner-light.svg   banner animado (retrato → logos)
    assets/whoami.svg                           tarjeta de terminal
    assets/radar-{dark,light}.svg               radar de habilidades
    assets/radar-langs-{dark,light}.svg         radar de lenguajes
    README.md                                   el perfil completo

Retrato: pon tu foto en assets/source/portrait.png (idealmente con fondo
quitado, de hombros hacia arriba). Si no existe, se usan tus iniciales.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import urllib.parse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
LOGOS = Path(__file__).resolve().parent / "logos"
PORTRAIT = ASSETS / "source" / "portrait.png"
CFG = json.loads((ROOT / "profile.config.json").read_text(encoding="utf-8"))


def _fill(v):
    """Sustituye {github} por tu usuario en cualquier texto del config."""
    if isinstance(v, str):
        return v.replace("{github}", CFG["github_username"])
    if isinstance(v, list):
        return [_fill(x) for x in v]
    if isinstance(v, dict):
        return {k: _fill(x) for k, x in v.items()}
    return v


CFG = _fill(CFG)

MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
SANS = "ui-sans-serif,-apple-system,Segoe UI,Helvetica,Arial,sans-serif"
SEED = 2718
TRAVELLERS = 800
HOLD_PARTICLES = 2200
MAX_PORTRAIT_DOTS = 18000
INTRO_S, TRANSITION_S, HOLD_S, PORTRAIT_S = 3.2, 1.3, 3.6, 3.0

BANNERS: dict[str, str] = {}

THEMES = {
    "dark": dict(bg="#0A0F1E", panel="#0D1628", panel2="#101B30", line="#25344C",
                 muted="#8291A8", text="#F0E6F0", dots="#F78CA0", chrome="#C9B1D9",
                 shadow="#02050B"),
    "light": dict(bg="#FDF0F3", panel="#FFFFFF", panel2="#FDE8EE", line="#F0C0CE",
                  muted="#9B7B8A", text="#2D1A24", dots="#E05F80", chrome="#7B5EA7",
                  shadow="#D4A0B0"),
}


def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def n1(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def t4(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".")


# ─────────────────────────────── banner ──────────────────────────────── #

def source_image() -> Image.Image:
    """Tu foto si existe; si no, tus iniciales grandes como fuente del dither."""
    if PORTRAIT.exists():
        src = Image.open(PORTRAIT).convert("RGBA")
        w, h = src.size
        frac, top_frac = CFG.get("portrait_crop", [0.62, 0.06])
        cw = int(w * frac)
        ch = int(cw * 340 / 300)
        left, top = (w - cw) // 2, int(h * top_frac)
        return src.crop((left, top, left + cw, min(h, top + ch))).resize((300, 340), Image.Resampling.LANCZOS)
    img = Image.new("RGBA", (300, 340), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    initials = "".join(p[0] for p in CFG["short_name"].split()[:2]).upper() or "?"
    text = initials if len(CFG["short_name"].split()) > 1 else CFG["short_name"][:3].upper()
    font = None
    for cand in ("DejaVuSansMono-Bold.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf"):
        try:
            font = ImageFont.truetype(cand, 150 if len(text) <= 2 else 110)
            break
        except OSError:
            continue
    font = font or ImageFont.load_default()
    box = d.textbbox((0, 0), text, font=font)
    x = (300 - (box[2] - box[0])) / 2 - box[0]
    y = (340 - (box[3] - box[1])) / 2 - box[1] - 20
    d.text((x, y), text, font=font, fill=(255, 255, 255, 255))
    d.text((150, 300), "</>", font=ImageFont.truetype(font.path, 40) if hasattr(font, "path") else font,
           fill=(255, 255, 255, 200), anchor="mm")
    return img


def dither(gray: np.ndarray) -> np.ndarray:
    """Floyd–Steinberg serpentina a 1 bit."""
    work = gray.astype(np.float32) / 255.0
    out = np.zeros(work.shape, dtype=bool)
    h, w = work.shape
    for y in range(h):
        ltr = y % 2 == 0
        step = 1 if ltr else -1
        for x in (range(w) if ltr else range(w - 1, -1, -1)):
            old = work[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = new > 0
            err = old - new
            if 0 <= x + step < w:
                work[y, x + step] += err * 7 / 16
            if y + 1 < h:
                if 0 <= x - step < w:
                    work[y + 1, x - step] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if 0 <= x + step < w:
                    work[y + 1, x + step] += err / 16
    return out


def silhouette_bits() -> np.ndarray:
    """Para fotos a contraluz: silueta oscura sobre fondo claro.

    El cuerpo se rellena con puntos suaves, las zonas iluminadas (cara) más
    densas, y el contorno se marca como un borde de luz.
    """
    from scipy import ndimage

    g = np.asarray(ImageOps.grayscale(source_image().convert("RGB")), dtype=np.float32)
    thr = float(CFG.get("silhouette_threshold", 34))
    # Se trabaja con bordes replicados para que el marco de la foto no cuente como contorno.
    pad = 8
    raw = np.pad(g < thr, pad, mode="edge")
    lab, count = ndimage.label(ndimage.binary_opening(raw, iterations=2))
    if not count:
        raise SystemExit("No se encontró la silueta: sube silhouette_threshold en el config.")
    sizes = ndimage.sum(raw, lab, range(1, count + 1))
    mask = lab == (1 + int(np.argmax(sizes)))
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=2))
    edge = (mask & ~ndimage.binary_erosion(mask, iterations=2))[pad:-pad, pad:-pad]
    mask = mask[pad:-pad, pad:-pad]
    lit = np.clip((g - 7) / 20, 0, 1)
    tone = np.where(mask, 0.08 + 0.55 * lit ** 0.8, 0)
    tone = np.maximum(tone, edge * 0.95)
    return dither(tone * 255)


def portrait_points(theme: str, rng: np.random.Generator) -> np.ndarray:
    if PORTRAIT.exists() and CFG.get("portrait_style") == "silhouette":
        ys, xs = np.where(silhouette_bits())
        pts = np.column_stack((94 + xs, 154 + ys)).astype(np.float32)
        if len(pts) > MAX_PORTRAIT_DOTS:
            pts = pts[rng.choice(len(pts), MAX_PORTRAIT_DOTS, replace=False)]
        return pts
    crop = source_image()
    alpha = np.asarray(crop.getchannel("A"), dtype=np.float32) / 255.0
    # Las iniciales (texto claro con alfa) se procesan igual en ambos temas.
    dark = theme == "dark" or not PORTRAIT.exists()
    base = Image.new("RGBA", crop.size, "black" if dark else "white")
    base.alpha_composite(crop)
    g = ImageOps.autocontrast(ImageOps.grayscale(base.convert("RGB")), cutoff=1)
    g = ImageEnhance.Contrast(g).enhance(1.35 if dark else 1.25)
    g = ImageEnhance.Brightness(g).enhance(1.05)
    g = g.filter(ImageFilter.UnsharpMask(radius=2.0, percent=155, threshold=1))
    bits = dither(np.asarray(g))
    active = bits if dark else ~bits
    if dark or not PORTRAIT.exists():
        active &= alpha > 0.08
    ys, xs = np.where(active)
    if not len(xs):
        raise SystemExit("El retrato no produjo puntos: usa una foto con más contraste o fondo quitado.")
    pts = np.column_stack((94 + xs, 154 + ys)).astype(np.float32)
    if len(pts) > MAX_PORTRAIT_DOTS:
        pts = pts[rng.choice(len(pts), MAX_PORTRAIT_DOTS, replace=False)]
    return pts


def load_logo(name: str) -> np.ndarray:
    path = next((p for p in (LOGOS / f"{name}.png", LOGOS / f"{name}.webp") if p.exists()), None)
    if path is None:
        raise SystemExit(f"Falta el logo {name}.png en scripts/logos/")
    icon = Image.open(path).convert("RGBA")
    icon.thumbnail((300, 300), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    canvas.alpha_composite(icon, ((400 - icon.width) // 2, (400 - icon.height) // 2))
    ys, xs = np.where(np.asarray(canvas.getchannel("A")) > 127)
    if not len(xs):
        raise SystemExit(f"El logo {name} no tiene píxeles visibles")
    return np.column_stack((109 + xs * 0.675, 188 + ys * 0.675)).astype(np.float32)


def sample(points: np.ndarray, rng: np.random.Generator, k: int) -> np.ndarray:
    return points[rng.choice(len(points), k, replace=len(points) < k)]


def transport(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    r, c = linear_sum_assignment(cdist(src, dst, "sqeuclidean"))
    out = np.empty_like(dst)
    out[r] = dst[c]
    return out


def runs_path(points: np.ndarray) -> str:
    """Une puntos horizontales contiguos en trazos cortos para ahorrar bytes."""
    if not len(points):
        return ""
    uniq = sorted({(int(x), int(y)) for x, y in np.rint(points).astype(int)}, key=lambda p: (p[1], p[0]))
    out, i = [], 0
    while i < len(uniq):
        x0, y = uniq[i]
        x1 = x0
        i += 1
        while i < len(uniq) and uniq[i][1] == y and uniq[i][0] <= x1 + 1:
            x1 = uniq[i][0]
            i += 1
        out.append(f"M{x0} {y}h{x1 - x0 + 1}")
    return "".join(out)


def dots_path(points: np.ndarray) -> str:
    uniq = sorted({(int(x), int(y)) for x, y in np.rint(points).astype(int)}, key=lambda p: (p[1], p[0]))
    return "".join(f"M{x} {y}h1" for x, y in uniq)


def render_banner(theme: str, portrait: np.ndarray, logos: dict[str, np.ndarray],
                  rng: np.random.Generator) -> str:
    t = THEMES[theme]
    W, H = 1180, 610
    n = min(TRAVELLERS, len(portrait))
    start = portrait[rng.choice(len(portrait), n, replace=False)]
    frames, times = [start, start], [0.0, PORTRAIT_S]
    cur = start
    holds = {}
    for name, pts in logos.items():
        cur = transport(cur, sample(pts, rng, n))
        frames += [cur, cur]
        times += [times[-1] + TRANSITION_S, times[-1] + TRANSITION_S + HOLD_S]
        holds[name] = sample(pts, rng, HOLD_PARTICLES)
    frames.append(start)
    times.append(times[-1] + TRANSITION_S)
    loop = times[-1]
    dur = t4(loop)
    kt = ";".join(t4(v / loop) for v in times)
    anim = f'begin="{INTRO_S}s" dur="{dur}s" repeatCount="indefinite" calcMode="linear" keyTimes="{kt}"'

    def txt(x, y, s, fill, size, extra=""):
        return f'<text x="{x}" y="{y}" fill="{fill}" font-family="{MONO}" font-size="{size}" {extra}>{s}</text>'

    user = esc(CFG["github_username"])
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        f'role="img" aria-labelledby="title desc"><title id="title">{esc(CFG["name"])} — {esc(CFG["role"])}</title>'
        '<desc id="desc">Perfil animado en terminal con retrato en puntos que se transforma en los logos del stack.</desc>',
        f'<defs><filter id="shadow" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="12" '
        f'stdDeviation="16" flood-color="{t["shadow"]}" flood-opacity=".28"/></filter>'
        '<clipPath id="vc"><rect x="49" y="124" width="390" height="414" rx="3"/></clipPath>'
        '<clipPath id="bc"><rect x="1100" y="533" width="8" height="18"/></clipPath></defs>',
        f'<rect width="{W}" height="{H}" rx="18" fill="{t["bg"]}"/>',
        f'<rect x="13" y="13" width="1154" height="584" rx="13" fill="{t["panel"]}" stroke="{t["line"]}" filter="url(#shadow)"/>',
        f'<path d="M13 62H1167" stroke="{t["line"]}"/>',
        '<circle cx="38" cy="38" r="6" fill="#FF5F57"/><circle cx="59" cy="38" r="6" fill="#FEBC2E"/>'
        '<circle cx="80" cy="38" r="6" fill="#28C840"/>',
        txt(590, 43, "vim profile.yml", t["muted"], 13, 'text-anchor="middle" letter-spacing=".4"'),
        f'<rect x="35" y="88" width="418" height="472" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>',
        f'<path d="M35 124H453" stroke="{t["line"]}"/>',
        txt(49, 111, "VISUAL.MAP", t["chrome"], 13, 'font-weight="700" letter-spacing="1.2"'),
        txt(438, 111, "300×340 / 1-BIT", t["muted"], 11, 'text-anchor="end"'),
        f'<path d="M49 141h12M49 141v12M439 141h-12M439 141v12M49 539h12M49 539v-12M439 539h-12M439 539v-12" '
        f'fill="none" stroke="{t["chrome"]}" opacity=".55"/>',
        '<g clip-path="url(#vc)" shape-rendering="crispEdges"><g>',
    ]

    # Retrato denso: se desvanece al empezar cada ciclo y vuelve al final.
    first_c = frames[2].mean(axis=0)
    band_ids = rng.integers(0, 80, size=len(portrait))
    noise = rng.normal(0, 4, size=(80, 2))
    k = len(frames)
    for b in range(80):
        pts = portrait[band_ids == b]
        if not len(pts):
            continue
        d = (first_c - pts.mean(axis=0)) * 0.18 + noise[b]
        pos = ["0 0", "0 0", f"{n1(d[0])} {n1(d[1])}", f"{n1(d[0])} {n1(d[1])}"] + ["0 0"] * (k - 4)
        op = [".94", ".94", "0", "0"] + ["0"] * (k - 5) + [".94"]
        p.append(f'<path d="{runs_path(pts)}" fill="none" stroke="{t["dots"]}" stroke-width="1" opacity=".94">'
                 f'<animateTransform attributeName="transform" type="translate" {anim} values="{";".join(pos)}"/>'
                 f'<animate attributeName="opacity" {anim} values="{";".join(op)}"/></path>')

    # Viajeros: cada punto recorre retrato → logo → logo → retrato.
    trav_op = ";".join(["0", "0"] + ["1"] * (k - 3) + ["0"])
    for i in range(n):
        vals = ";".join(f"{n1(f[i, 0])} {n1(f[i, 1])}" for f in frames)
        p.append(f'<path d="M-.65-.65h1.3v1.3h-1.3z" fill="{t["dots"]}">'
                 f'<animateTransform attributeName="transform" type="translate" {anim} values="{vals}"/>'
                 f'<animate attributeName="opacity" {anim} values="{trav_op}"/></path>')

    # Nube densa que "rellena" cada logo mientras se mantiene.
    for idx, pts in enumerate(holds.values()):
        vis = ["0"] * k
        vis[idx * 2 + 2] = vis[idx * 2 + 3] = ".82"
        p.append(f'<path d="{dots_path(pts)}" fill="none" stroke="{t["dots"]}" stroke-width="1" opacity="0">'
                 f'<animate attributeName="opacity" {anim} values="{";".join(vis)}"/></path>')
    p.append("</g>")

    # Intro: el retrato aparece por grupos aleatorios.
    ids = rng.integers(0, 60, size=len(portrait))
    starts = np.empty(60)
    starts[rng.permutation(60)] = np.linspace(0.05, 1.2, 60)
    for g in range(60):
        pts = portrait[ids == g]
        if len(pts):
            p.append(f'<path d="{runs_path(pts)}" fill="none" stroke="{t["dots"]}" stroke-width="1" opacity="0">'
                     f'<animate attributeName="opacity" begin="{n1(starts[g])}s" dur=".8s" values="0;1" fill="freeze"/>'
                     f'<animate attributeName="opacity" begin="3.08s" dur=".12s" values="1;0" fill="freeze"/></path>')
    p.append("</g>")
    p.append(txt(58, 551, f"PTS {len(portrait):05d} · FS/SERPENTINE", t["muted"], 10))

    # Panel derecho: profile.yml en vim.
    badge = f"@{CFG['github_username']}"
    bw = max(110, int(len(badge) * 8.4 + 30))
    p += [
        f'<rect x="474" y="88" width="672" height="472" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>',
        f'<path d="M474 124H1146" stroke="{t["line"]}"/>',
        txt(490, 111, "profile.yml", t["chrome"], 13, 'font-weight="700" letter-spacing=".5"'),
        txt(586, 111, "[YAML]", t["muted"], 11),
        f'<rect x="{1128 - bw}" y="94" width="{bw}" height="24" rx="12" fill="{t["chrome"]}" opacity=".16" stroke="{t["chrome"]}"/>',
        txt(1128 - bw / 2, 111, esc(badge), t["chrome"], 13, 'text-anchor="middle" font-weight="700"'),
    ]
    rows = CFG["banner_yaml"][:17]
    y = 148.0
    total_bytes = 0
    for i, (indent, key, val) in enumerate(rows, 1):
        if indent == 0:
            body, x = f'<tspan fill="{t["chrome"]}" font-weight="700">{esc(key)}:</tspan>', 525
        else:
            body, x = (f'<tspan fill="{t["dots"]}">{esc(key)}: </tspan>'
                       f'<tspan fill="{t["text"]}">{esc(val)}</tspan>'), 542
        total_bytes += len(f"{'  ' * indent}{key}: {val}\n".encode())
        p.append(txt(506, n1(y), f"{i:2d}", t["muted"], 13, 'text-anchor="end" opacity=".45"'))
        p.append(f'<text x="{x}" y="{n1(y)}" font-family="{MONO}" font-size="13">{body}</text>')
        y += 21.5
    nrows = len(rows)
    p += [
        f'<path d="M474 526H1146" stroke="{t["line"]}"/>',
        f'<rect x="475" y="527" width="670" height="32" fill="{t["panel"]}"/>',
        f'<rect x="485" y="533" width="72" height="20" rx="3" fill="{t["dots"]}"/>',
        txt(521, 547, "NORMAL", t["bg"], 11, 'text-anchor="middle" font-weight="700"'),
        txt(569, 547, "profile.yml", t["text"], 12, 'font-weight="600"'),
        txt(740, 547, "[utf-8]", t["muted"], 11),
        txt(1134, 547, f"{nrows}L, {total_bytes}B  100%  {nrows}:1", t["muted"], 11, 'text-anchor="end"'),
        "</svg>",
    ]
    return "".join(p)


# ─────────────────────────────── whoami ──────────────────────────────── #

def render_whoami() -> str:
    def tx(x, y, s, fill, size, weight=""):
        w = f' font-weight="{weight}"' if weight else ""
        return f'<text x="{x}" y="{y}" fill="{fill}" font-family="{MONO}" font-size="{size}"{w}>{esc(s)}</text>'

    handle = CFG["short_name"].lower().replace(" ", "_")
    role_x = 48 + len(handle) * 10.9 + 14
    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 390" width="960" height="390" role="img" aria-labelledby="t d">',
        f'<title id="t">{esc(CFG["name"])} — {esc(CFG["role"])}</title>',
        '<desc id="d">Tarjeta de terminal estilo city-pop con ubicación, misión y enfoque técnico.</desc>',
        '<defs><linearGradient id="sky" x1="0" x2="1" y1="0" y2="1"><stop stop-color="#c7a4f5"/>'
        '<stop offset="1" stop-color="#77c8f0"/></linearGradient>'
        '<filter id="glow" x="-20%" y="-30%" width="140%" height="160%"><feGaussianBlur stdDeviation="2.2" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>',
        '<rect width="960" height="390" rx="18" fill="#17171c"/>',
        '<rect x="9" y="9" width="942" height="372" rx="13" fill="none" stroke="url(#sky)" stroke-width="2"/>',
        '<circle cx="35" cy="35" r="5" fill="#f3a0b5"/><circle cx="54" cy="35" r="5" fill="#f1d09a"/>'
        '<circle cx="73" cy="35" r="5" fill="#9de4d1"/>',
        tx(98, 41, f"{CFG['github_username']}  ·  profile shell  ·  online", "#8f91a8", 15),
        '<path d="M29 58H931" stroke="#343441"/>',
        '<rect x="29" y="78" width="594" height="271" rx="9" fill="#1d1d23" stroke="#b99af0" stroke-width="1.5"/>',
        tx(48, 111, "❯ whoami", "#76d8d2", 20, "700"),
        tx(48, 143, handle, "#b8b9ca", 18),
        tx(n1(role_x), 143, "—", "#8f91a8", 18),
        tx(n1(role_x + 25), 143, CFG["role"], "#f3d29b", 18),
        '<path d="M48 166H604" stroke="#343441"/>',
        tx(48, 196, "context:", "#8f91a8", 17),
        tx(166, 196, f"{CFG['location']}  {CFG.get('flag', '')}".strip(), "#c7a4f5", 17),
        tx(48, 225, "mission:", "#8f91a8", 17),
        tx(166, 225, CFG["mission"], "#9de4d1", 17),
        tx(48, 267, "❯ ls expertise/", "#76d8d2", 20, "700"),
    ]
    for i, (folder, desc) in enumerate(CFG["whoami_expertise"][:2]):
        y = 299 + i * 27
        out += [tx(48, y, folder, "#f3a0b5", 16), tx(214, y, desc, "#b8b9ca", 16)]
    out += [
        '<rect x="651" y="78" width="280" height="271" rx="9" fill="#1d1d23" stroke="#b99af0" stroke-width="1.5"/>',
        tx(674, 111, "city-pop runtime", "#f3d29b", 17),
        '<circle cx="791" cy="203" r="56" fill="#f3a0b5" opacity=".95"/>',
        '<path d="M674 202H908M674 216H908M674 230H908M674 244H908M674 258H908" stroke="#1d1d23" stroke-width="5" opacity=".9"/>',
        '<path d="M674 277C711 247 746 303 785 272S861 249 908 281" fill="none" stroke="#76d8d2" stroke-width="3" filter="url(#glow)"/>',
        '<path d="M674 294C716 264 746 320 789 289S862 270 908 298" fill="none" stroke="#c7a4f5" stroke-width="2"/>',
        tx(674, 326, CFG["status_line"], "#8f91a8", 14),
        "</svg>",
    ]
    return "".join(out)


# ─────────────────────────────── radar ───────────────────────────────── #

RADAR = {
    "dark": dict(grid="#30363d", spoke="#21262d", label="#c9d1d9", title="#e6edf3", fill="#f78ca0", stroke="#f78ca0"),
    "light": dict(grid="#d0d7de", spoke="#e6eaef", label="#1f2328", title="#1f2328", fill="#e05f80", stroke="#e05f80"),
}


def render_radar(title: str, axes: list, theme: str, size: int = 440, rings: int = 4) -> str:
    c = RADAR[theme]
    n = len(axes)
    r, gap = size / 2 - 8, 20
    vals = [max(0.0, min(100.0, float(v))) for _, v in axes]

    def ring(rad):
        return [(rad * math.cos(-math.pi / 2 + i * 2 * math.pi / n),
                 rad * math.sin(-math.pi / 2 + i * 2 * math.pi / n)) for i in range(n)]

    outer = ring(r)
    labels = []
    minx, maxx, miny, maxy = -r, r, -r, r
    for i, (lab, _) in enumerate(axes):
        a = -math.pi / 2 + i * 2 * math.pi / n
        cs, sn = math.cos(a), math.sin(a)
        lx, ly = (r + gap) * cs, (r + gap) * sn
        anchor = "middle" if abs(cs) < 0.25 else ("start" if cs > 0 else "end")
        ly += 4 if abs(sn) < 0.25 else (14 if sn > 0 else -5)
        w = len(lab) * 13 * 0.62
        x0 = lx if anchor == "start" else (lx - w if anchor == "end" else lx - w / 2)
        minx, maxx = min(minx, x0), max(maxx, x0 + w)
        miny, maxy = min(miny, ly - 13), max(maxy, ly + 4)
        labels.append((lx, ly, anchor, lab))
    pad, th = 10, 29
    W = round(maxx - minx + 2 * pad)
    H = round(maxy - miny + 2 * pad + th)
    ox, oy = -minx + pad, -miny + pad + th
    need = round(len(title) * 15 * 0.62 + 2 * pad)
    if need > W:
        ox += (need - W) / 2
        W = need
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-label="{esc(title)}" font-family="{SANS}">',
         f'<text x="{W / 2:.1f}" y="25" text-anchor="middle" font-size="15" font-weight="700" fill="{c["title"]}">{esc(title)}</text>',
         f'<g transform="translate({ox:.1f},{oy:.1f})">']
    for k in range(rings, 0, -1):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in ring(r * k / rings))
        s.append(f'<polygon points="{pts}" fill="none" stroke="{c["grid"]}" opacity="{0.35 + 0.5 * k / rings:.2f}"/>')
    s += [f'<line x1="0" y1="0" x2="{x:.1f}" y2="{y:.1f}" stroke="{c["spoke"]}"/>' for x, y in outer]
    shape = [(x * v / 100, y * v / 100) for (x, y), v in zip(outer, vals)]
    s.append('<g><animateTransform attributeName="transform" type="scale" values="0.04;1" dur="1.1s" '
             'calcMode="spline" keyTimes="0;1" keySplines="0.22 1 0.36 1" fill="freeze"/>')
    s.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in shape)}" fill="{c["fill"]}" '
             f'fill-opacity="0.22" stroke="{c["stroke"]}" stroke-width="2.5" stroke-linejoin="round"/>')
    s += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="{c["fill"]}"/>' for x, y in shape]
    s.append("</g>")
    s += [f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{a}" font-size="13" font-weight="600" fill="{c["label"]}">{esc(l)}</text>'
          for x, y, a, l in labels]
    s.append("</g></svg>")
    return "".join(s)


# ─────────────────────────────── README ──────────────────────────────── #

def render_readme() -> str:
    u = CFG["github_username"]
    lines = urllib.parse.quote(";".join(CFG["typing_lines"]), safe="")
    typing = ("https://readme-typing-svg.demolab.com?font=JetBrains+Mono&amp;weight=700&amp;size=26"
              "&amp;duration=2600&amp;pause=900&amp;color=F78CA0&amp;center=true&amp;vCenter=true"
              f"&amp;width=900&amp;lines={lines}")
    stack = CFG["tech_stack"]
    cells = []
    for i, cat in enumerate(stack):
        branch = "╰─" if i == len(stack) - 1 else "├─"
        cells.append(
            f'      <td width="50%" valign="top"><code>{branch} {esc(cat["key"])}:</code><br><br>\n'
            f'        <img src="https://skillicons.dev/icons?i={cat["icons"]}" alt="{esc(cat["names"])}"><br>\n'
            f'        <sub><code>{esc(cat["names"])}</code></sub>\n      </td>')
    rows = []
    for i in range(0, len(cells), 2):
        pair = cells[i:i + 2]
        if len(pair) == 1:
            pair[0] = pair[0].replace('<td width="50%"', '<td colspan="2"')
        rows.append("    <tr>\n" + "\n".join(pair) + "\n    </tr>")
    socials = "&nbsp;&nbsp;\n".join(
        f'<a href="{s["url"]}">\n  <img src="https://img.shields.io/badge/{urllib.parse.quote(s["label"])}-{s["color"]}'
        f'?style=for-the-badge&logo={s["logo"]}&logoColor={s["logoColor"]}" alt="{esc(s["label"])}">\n</a>'
        for s in CFG["socials"])
    return f"""<div align="center">

<!-- BANNER ANIMADO (generado por scripts/generate.py) -->
<a href="https://github.com/{u}">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/{BANNERS['dark']}">
    <source media="(prefers-color-scheme: light)" srcset="assets/{BANNERS['light']}">
    <img src="assets/{BANNERS['light']}" width="960" alt="Perfil de {esc(CFG['name'])}, {esc(CFG['role'])}">
  </picture>
</a>

<br>

<a href="https://github.com/{u}">
  <img src="{typing}" alt="Texto animado del perfil">
</a>

<img src="https://komarev.com/ghpvc/?username={u}&style=flat&color=f78ca0&label=profile+views" alt="profile views">

</div>

---

## `$ whoami`

<p align="center">
  <img src="assets/whoami.svg" width="960" alt="Terminal city-pop con el perfil de {esc(CFG['name'])}">
</p>

<br>

<div align="center">

## `$ cat tech-stack.yaml`

<table border="1" cellpadding="14" bgcolor="#17171c">
  <thead>
    <tr>
      <th colspan="2" align="left"><code>{esc(u)}:~$ cat tech-stack.yaml</code></th>
    </tr>
  </thead>
  <tbody>
{chr(10).join(rows)}
  </tbody>
  <tfoot>
    <tr>
      <td colspan="2"><code>status: building&nbsp;&nbsp;·&nbsp;&nbsp;environment: production</code></td>
    </tr>
  </tfoot>
</table>

</div>

---

## `$ npm run stats --all`

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/radar-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/radar-light.svg">
    <img src="assets/radar-light.svg" width="390" alt="Radar de habilidades">
  </picture>&nbsp;
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/radar-langs-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/radar-langs-light.svg">
    <img src="assets/radar-langs-light.svg" width="370" alt="Radar de lenguajes">
  </picture>
</p>

<p align="center"><sub><code>signals: skill_radar · language_stack_radar · status: healthy</code></sub></p>

---

## `$ connect --socials`

<div align="center">

{socials}

</div>

<br>
<br>

<div align="center">
<sub>{esc(CFG['footer'])} · @{esc(u)}</sub>
</div>
"""


# ─────────────────────────────── main ────────────────────────────────── #

def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    logos = {name: load_logo(name) for name in CFG["banner_logos"]}
    # El nombre del banner lleva un hash del contenido: GitHub cachea las imágenes
    # por URL, así que un nombre nuevo obliga a mostrar siempre la versión actual.
    for old in ASSETS.glob("banner-*.svg"):
        old.unlink()
    for i, theme in enumerate(THEMES):
        rng = np.random.default_rng(SEED + i)
        portrait = portrait_points(theme, rng)
        svg = render_banner(theme, portrait, logos, rng)
        digest = hashlib.sha1(svg.encode()).hexdigest()[:8]
        out = ASSETS / f"banner-{theme}.{digest}.svg"
        out.write_text(svg, encoding="utf-8")
        BANNERS[theme] = out.name
        print(f"{out.relative_to(ROOT)}: {out.stat().st_size / 1024:.0f} KiB, {len(portrait):,} puntos")
    (ASSETS / "whoami.svg").write_text(render_whoami(), encoding="utf-8")
    for key, stem in (("skills_radar", "radar"), ("langs_radar", "radar-langs")):
        data = CFG[key]
        for theme in RADAR:
            (ASSETS / f"{stem}-{theme}.svg").write_text(render_radar(data["title"], data["axes"], theme), encoding="utf-8")
    (ROOT / "README.md").write_text(render_readme(), encoding="utf-8")
    print("whoami.svg, radares y README.md generados")
    if not PORTRAIT.exists():
        print("Aviso: no hay assets/source/portrait.png, el banner usa tus iniciales.")


if __name__ == "__main__":
    main()
