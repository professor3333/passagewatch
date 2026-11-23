"""The README's figures, written as SVG from committed files.

Every number drawn comes from a committed record: the release's one-time test evaluation
(``releases/evaluations/<release>-test.json``) and the CFC authors' published tracks as
scored by CFC's official evaluator (``tests/regression/cfc_official_nmae_eccv22.json``).
``scripts/make_readme_figures.py --check`` fails when a committed figure no longer matches
what this module writes, so the chart cannot drift from the results.

Most figures come in a light and a dark variant (GitHub picks one with ``<picture>``); the
banner has its own dark background and works on both. Text uses the system font stack,
so it renders like the surrounding page.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"

TEST_LOCATIONS = ("elwha", "kenai-channel", "kenai-rightbank", "nushagak")

Point = tuple[float, float]


@dataclass(frozen=True)
class Theme:
    """Colors for one GitHub theme. Text colors are GitHub's own, so figures blend in."""

    name: str
    text: str
    muted: str
    grid: str
    axis: str
    card: str
    card_border: str
    accent: str
    accent_wash: str
    human: str
    human_wash: str
    neutral_wash: str
    # Categorical series, in fixed order (validated for adjacent-pair CVD separation).
    series: tuple[str, str, str, str]


LIGHT = Theme(
    name="light",
    text="#1f2328",
    muted="#59636e",
    grid="#e6eaef",
    axis="#d1d9e0",
    card="#f6f8fa",
    card_border="#d1d9e0",
    accent="#0969da",
    accent_wash="#ddf4ff",
    human="#8250df",
    human_wash="#fbefff",
    neutral_wash="#eff2f5",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
)

DARK = Theme(
    name="dark",
    text="#f0f6fc",
    muted="#9198a1",
    grid="#262c36",
    axis="#3d444d",
    card="#151b23",
    card_border="#3d444d",
    accent="#4493f8",
    accent_wash="#0d2a4d",
    human="#ab7df8",
    human_wash="#2a1f45",
    neutral_wash="#212830",
    series=("#3987e5", "#d95926", "#199e70", "#c98500"),
)

THEMES = (LIGHT, DARK)

# Sonar imagery is dark in both themes, as in the review page.
SONAR_BG = "#0b1119"
SONAR_FAN = "#18212c"
SONAR_LINE = "#e6edf3"
TRACK = "#58a6ff"


@dataclass(frozen=True)
class LocationResult:
    name: str
    clips: int
    passages: int
    nmae: dict[str, float]  # system key -> nMAE


@dataclass(frozen=True)
class BenchmarkResults:
    release: str
    locations: tuple[LocationResult, ...]
    macro: dict[str, float]

    @property
    def clips(self) -> int:
        return sum(loc.clips for loc in self.locations)

    @property
    def passages(self) -> int:
        return sum(loc.passages for loc in self.locations)


# Drawn in this order, with series colors in this order.
SYSTEMS: tuple[tuple[str, str], ...] = (
    ("release", "PassageWatch {release}"),
    ("classical", "Tuned classical baseline"),
    ("cfc_baseline", "CFC Baseline (published)"),
    ("cfc_baseline_pp", "CFC Baseline++ (published)"),
)


def load_test_results(evaluation: Path, cfc_reference: Path) -> BenchmarkResults:
    """Per-location nMAE of the release, the classical baseline and CFC's two methods."""
    report = json.loads(evaluation.read_text())
    reference = json.loads(cfc_reference.read_text())["results"]
    locations = []
    for name in TEST_LOCATIONS:
        entry = report["locations"][name]
        nmae = {
            "release": float(entry["release"]["nmae"]),
            "classical": float(entry["classical"]["nmae"]),
        }
        for key, tracker in (("cfc_baseline", "baseline"), ("cfc_baseline_pp", "baseline++")):
            clips = reference[tracker][name]
            numerator = sum(int(v[0]) for v in clips.values())
            denominator = sum(int(v[1]) for v in clips.values())
            nmae[key] = numerator / denominator
        locations.append(LocationResult(name, int(entry["clips"]), int(entry["passages"]), nmae))
    macro = {key: sum(loc.nmae[key] for loc in locations) / len(locations) for key, _ in SYSTEMS}
    return BenchmarkResults(str(report["release"]), tuple(locations), macro)


# --- SVG helpers -------------------------------------------------------------------------


def _n(value: float) -> str:
    """A coordinate with at most two decimals and no trailing zeros (stable output)."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _text(
    x: float,
    y: float,
    content: str,
    *,
    size: float,
    fill: str,
    weight: int = 400,
    anchor: str = "start",
    extra: str = "",
) -> str:
    weight_attr = f' font-weight="{weight}"' if weight != 400 else ""
    anchor_attr = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return (
        f'<text x="{_n(x)}" y="{_n(y)}" font-size="{_n(size)}" fill="{fill}"'
        f"{weight_attr}{anchor_attr}{extra}>{_esc(content)}</text>"
    )


def _svg(width: int, height: int, title: str, desc: str, body: Sequence[str]) -> str:
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc" '
        f'font-family="{FONT}">'
    )
    lines = [
        head,
        f'<title id="title">{_esc(title)}</title>',
        f'<desc id="desc">{_esc(desc)}</desc>',
        *body,
        "</svg>",
    ]
    return "\n".join(lines) + "\n"


_NARROW = frozenset("iljtfrI.,:;'|!·() ")
_WIDE = frozenset("mwMW@%")


def _text_width(text: str, size: float) -> float:
    """A width estimate, slightly generous so wider system fonts (Linux) still fit."""
    units = 0.0
    for char in text:
        if char in _NARROW:
            units += 0.32
        elif char in _WIDE:
            units += 0.86
        elif char.isupper():
            units += 0.68
        else:
            units += 0.57
    return units * size * 1.06


def _pill(x: float, y: float, label: str, *, size: float, fill: str, ink: str) -> str:
    """A rounded label whose left edge is ``x`` and vertical center is ``y``."""
    width = _text_width(label, size) + size * 1.4
    height = size * 1.9
    return (
        f'<rect x="{_n(x)}" y="{_n(y - height / 2)}" width="{_n(width)}" '
        f'height="{_n(height)}" rx="{_n(height / 2)}" fill="{fill}"/>'
        + _text(
            x + width / 2, y + size * 0.35, label, size=size, fill=ink, weight=600, anchor="middle"
        )
    )


def _catmull_rom(points: Sequence[Point], per_segment: int = 10) -> list[Point]:
    """A smooth curve through ``points``, sampled evenly in parameter."""
    padded = [points[0], *points, points[-1]]
    out: list[Point] = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        for step in range(per_segment):
            t = step / per_segment
            t2, t3 = t * t, t * t * t
            out.append(
                tuple(  # type: ignore[arg-type]
                    0.5
                    * (
                        2 * p1[k]
                        + (-p0[k] + p2[k]) * t
                        + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2
                        + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3
                    )
                    for k in (0, 1)
                )
            )
    out.append(points[-1])
    return out


def _arrowhead(tip_from: Point, tip: Point, size: float, fill: str) -> str:
    dx, dy = tip[0] - tip_from[0], tip[1] - tip_from[1]
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    front = (tip[0] + ux * size * 0.6, tip[1] + uy * size * 0.6)
    back = (tip[0] - ux * size * 0.7, tip[1] - uy * size * 0.7)
    left = (back[0] + nx * size * 0.6, back[1] + ny * size * 0.6)
    right = (back[0] - nx * size * 0.6, back[1] - ny * size * 0.6)
    pts = " ".join(f"{_n(px)},{_n(py)}" for px, py in (front, left, right))
    return f'<polygon points="{pts}" fill="{fill}"/>'


def _track(points: Sequence[Point], *, color: str, background: str, dot: float = 3.0) -> str:
    """A trajectory: a faint path, fading detections, a ring at the start, an arrow at the end."""
    samples = _catmull_rom(points)
    path = "M" + " L".join(f"{_n(x)} {_n(y)}" for x, y in samples)
    parts = [
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" opacity="0.35"/>'
    ]
    shown = samples[3:-3:3]
    for i, (x, y) in enumerate(shown):
        opacity = 0.3 + 0.7 * (i + 1) / len(shown)
        parts.append(
            f'<circle cx="{_n(x)}" cy="{_n(y)}" r="{_n(dot)}" fill="{color}" '
            f'opacity="{_n(opacity)}"/>'
        )
    sx, sy = samples[0]
    parts.append(
        f'<circle cx="{_n(sx)}" cy="{_n(sy)}" r="{_n(dot * 2)}" fill="{background}" '
        f'stroke="{color}" stroke-width="2.5"/>'
    )
    parts.append(_arrowhead(samples[-3], samples[-1], dot * 5, color))
    return "\n".join(parts)


# --- Banner --------------------------------------------------------------------------------


def banner_svg() -> str:
    """Title, tagline and a stylized sonar view with a fish crossing the counting line."""
    width, height = 1280, 320
    apex = (1010.0, -250.0)
    inner, outer, half = 275.0, 560.0, math.radians(21)

    def polar(radius: float, angle: float) -> Point:
        return (apex[0] + radius * math.sin(angle), apex[1] + radius * math.cos(angle))

    a, b = polar(inner, -half), polar(inner, half)
    c, d = polar(outer, half), polar(outer, -half)
    fan = (
        f"M{_n(a[0])} {_n(a[1])} A{_n(inner)} {_n(inner)} 0 0 0 {_n(b[0])} {_n(b[1])} "
        f"L{_n(c[0])} {_n(c[1])} A{_n(outer)} {_n(outer)} 0 0 1 {_n(d[0])} {_n(d[1])} Z"
    )
    top_y, bottom_y = apex[1] + inner, apex[1] + outer

    crossing = [(872.0, 246.0), (930.0, 236.0), (990.0, 233.0), (1040.0, 226.0), (1076.0, 222.0)]
    upper = [(905.0, 112.0), (930.0, 121.0), (955.0, 133.0)]

    def fish(x: float, y: float, angle: float, label: str) -> str:
        return (
            f'<g transform="translate({_n(x)} {_n(y)}) rotate({_n(angle)})">'
            '<ellipse rx="19" ry="5.5" fill="#f2f6fa" opacity="0.92" filter="url(#soft)"/>'
            '<path d="M-19 0 L-27 -5 L-25 0 L-27 5 Z" fill="#f2f6fa" opacity="0.7" '
            'filter="url(#soft)"/>'
            "</g>"
            f'<rect x="{_n(x - 32)}" y="{_n(y - 13)}" width="62" height="26" rx="3" '
            f'fill="none" stroke="{TRACK}" stroke-width="2"/>'
            + _text(x - 32, y - 19, label, size=14, fill=TRACK, weight=600)
        )

    def track_dots(points: Sequence[Point]) -> str:
        samples = _catmull_rom(points, per_segment=6)[:-4]
        out = []
        for i, (x, y) in enumerate(samples[::2]):
            opacity = 0.15 + 0.6 * i / max(1, len(samples[::2]) - 1)
            out.append(
                f'<circle cx="{_n(x)}" cy="{_n(y)}" r="2.6" fill="{TRACK}" '
                f'opacity="{_n(opacity)}"/>'
            )
        return "\n".join(out)

    pills = ["CFC benchmark", "CPU · ONNX Runtime", "Human review", "Audit trail"]
    pill_parts = []
    x = 76.0
    for label in pills:
        w = _text_width(label, 15) + 28
        pill_parts.append(
            f'<rect x="{_n(x)}" y="252" width="{_n(w)}" height="32" rx="16" '
            'fill="#ffffff" fill-opacity="0.06" stroke="#ffffff" stroke-opacity="0.16"/>'
        )
        pill_parts.append(
            _text(x + w / 2, 273, label, size=15, fill="#b6c4d4", weight=500, anchor="middle")
        )
        x += w + 10

    body = [
        "<defs>",
        '<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#0a1424"/><stop offset="1" stop-color="#0e2440"/>'
        "</linearGradient>",
        '<radialGradient id="glow" cx="0.79" cy="0.62" r="0.42">'
        '<stop offset="0" stop-color="#1f6feb" stop-opacity="0.35"/>'
        '<stop offset="1" stop-color="#1f6feb" stop-opacity="0"/></radialGradient>',
        '<radialGradient id="fanfill" cx="1010" cy="-250" r="560" '
        'gradientUnits="userSpaceOnUse">'
        '<stop offset="0.45" stop-color="#2a3442"/><stop offset="1" stop-color="#151c26"/>'
        "</radialGradient>",
        '<filter id="speckle" x="0" y="0" width="100%" height="100%">'
        '<feTurbulence type="fractalNoise" baseFrequency="0.07 0.22" numOctaves="2" '
        'seed="11"/>'
        '<feColorMatrix values="0 0 0 0 0.86  0 0 0 0 0.9  0 0 0 0 0.94  2.2 0 0 0 -1.05"/>'
        "</filter>",
        '<filter id="soft" x="-50%" y="-50%" width="200%" height="200%">'
        '<feGaussianBlur stdDeviation="1.4"/></filter>',
        f'<clipPath id="fan"><path d="{fan}"/></clipPath>',
        '<clipPath id="card"><rect width="1280" height="320" rx="16"/></clipPath>',
        '<linearGradient id="mark" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#2f81f7"/><stop offset="1" stop-color="#1158c7"/>'
        "</linearGradient>",
        "</defs>",
        '<g clip-path="url(#card)">',
        '<rect width="1280" height="320" fill="url(#bg)"/>',
        '<rect width="1280" height="320" fill="url(#glow)"/>',
        f'<path d="{fan}" fill="url(#fanfill)"/>',
        '<g clip-path="url(#fan)" opacity="0.32">'
        '<rect x="760" y="0" width="500" height="320" filter="url(#speckle)"/></g>',
        f'<path d="{fan}" fill="none" stroke="#ffffff" stroke-opacity="0.12"/>',
        f'<line x1="{_n(apex[0])}" y1="{_n(top_y)}" x2="{_n(apex[0])}" y2="{_n(bottom_y)}" '
        f'stroke="{SONAR_LINE}" stroke-opacity="0.8" stroke-width="2" '
        'stroke-dasharray="7 7"/>',
        track_dots(crossing),
        track_dots(upper),
        fish(1076, 222, -6, "#7"),
        fish(955, 133, 24, "#8"),
        _pill(1118, 186, "+1 → right", size=15, fill="#1f6feb", ink="#ffffff"),
        "</g>",
        '<rect x="0.5" y="0.5" width="1279" height="319" rx="15.5" fill="none" '
        'stroke="#ffffff" stroke-opacity="0.1"/>',
        # Logo mark: a sonar wedge with a fish crossing a dashed line.
        '<g transform="translate(76 76)">',
        '<rect width="60" height="60" rx="15" fill="url(#mark)"/>',
        '<path d="M30 9 L13 46 Q30 54 47 46 Z" fill="#ffffff" fill-opacity="0.18"/>',
        '<line x1="30" y1="10" x2="30" y2="52" stroke="#ffffff" stroke-width="2.5" '
        'stroke-dasharray="4 3.5" stroke-linecap="round"/>',
        '<path d="M17 35 Q30 30 42 31" fill="none" stroke="#ffffff" stroke-width="3.5" '
        'stroke-linecap="round"/>',
        '<path d="M38 26 L45 31 L38 36" fill="none" stroke="#ffffff" stroke-width="3.5" '
        'stroke-linecap="round" stroke-linejoin="round"/>',
        "</g>",
        _text(
            154,
            128,
            "PassageWatch",
            size=60,
            fill="#f0f6fc",
            weight=700,
            extra=' letter-spacing="-1"',
        ),
        _text(
            78, 184, "Directional fish-passage counts from sonar video,", size=26, fill="#d1dbe6"
        ),
        _text(78, 220, "with the evidence to check them.", size=26, fill="#d1dbe6"),
        *pill_parts,
    ]
    return _svg(
        width,
        height,
        "PassageWatch",
        "Directional fish-passage counts from sonar video, with the evidence to check them. "
        "A stylized sonar view shows a tracked fish crossing the dashed counting line from "
        "left to right and being counted as one passage to the right.",
        body,
    )


# --- Pipeline ------------------------------------------------------------------------------

# 24x24 line icons.
_ICONS = {
    "sonar": '<path d="M12 3 L4.5 18.5 Q12 22.5 19.5 18.5 Z"/>'
    '<path d="M8.2 12.6 Q12 14.6 15.8 12.6"/><path d="M6.6 15.8 Q12 18.6 17.4 15.8"/>',
    "detect": '<rect x="3" y="6" width="18" height="12" rx="1.5" stroke-dasharray="3 2.2"/>'
    '<ellipse cx="13" cy="12" rx="4.6" ry="2"/><path d="M8.4 12 L5.8 10.2 V13.8 Z"/>',
    "track": '<path d="M3.5 17.5 C8 17.5 10 7 20.5 6.5" stroke-dasharray="0.1 4.4" '
    'stroke-width="3"/><circle cx="20.5" cy="6.5" r="2.2"/>'
    '<circle cx="3.5" cy="17.5" r="1.6"/>',
    "count": '<path d="M12 3 V21" stroke-dasharray="2.4 2.4"/><path d="M4 12 H19"/>'
    '<path d="M15.5 8.5 L19 12 L15.5 15.5"/>',
    "review": '<circle cx="12" cy="12" r="8.5"/><path d="M8 12.4 L10.8 15.2 L16.2 9.4"/>',
    "export": '<path d="M6 3 H14 L18 7 V21 H6 Z"/><path d="M14 3 V7 H18"/>'
    '<path d="M12 10.5 V17"/><path d="M9.2 14.4 L12 17.2 L14.8 14.4"/>',
}

_STAGES = (
    ("sonar", "Sonar video", "frames or video", False),
    ("detect", "Detect", "YOLOX-Tiny · ONNX", False),
    ("track", "Track", "Kalman tracker", False),
    ("count", "Count", "start/end rule", False),
    ("review", "Review", "queue + audits", True),
    ("export", "Export", "CSV · JSON", True),
)


def pipeline_svg(theme: Theme) -> str:
    """Six stages, grouped into what the worker does and what the technician does."""
    width, height = 1146, 222
    node_w, node_h, gap, left, top = 166.0, 138.0, 22.0, 20.0, 64.0
    body: list[str] = []

    def node_x(i: int) -> float:
        return left + i * (node_w + gap)

    for first, last, label, color in (
        (0, 3, "Automatic · one inference worker on CPU", theme.accent),
        (4, 5, "Technician · review page", theme.human),
    ):
        x0, x1 = node_x(first) + 6, node_x(last) + node_w - 6
        body.append(
            f'<path d="M{_n(x0)} 50 V42 H{_n(x1)} V50" fill="none" stroke="{color}" '
            'stroke-width="1.5" stroke-opacity="0.7"/>'
        )
        body.append(
            _text((x0 + x1) / 2, 31, label, size=15, fill=color, weight=600, anchor="middle")
        )

    for i, (icon, title, subtitle, human) in enumerate(_STAGES):
        x = node_x(i)
        color = theme.human if human else theme.accent
        wash = theme.human_wash if human else theme.accent_wash
        body.append(
            f'<rect x="{_n(x)}" y="{_n(top)}" width="{_n(node_w)}" height="{_n(node_h)}" '
            f'rx="14" fill="{theme.card}" stroke="{theme.card_border}"/>'
        )
        cx = x + node_w / 2
        body.append(f'<circle cx="{_n(cx)}" cy="{_n(top + 38)}" r="24" fill="{wash}"/>')
        body.append(
            f'<g transform="translate({_n(cx - 15)} {_n(top + 23)}) scale(1.25)" fill="none" '
            f'stroke="{color}" stroke-width="1.8" stroke-linecap="round" '
            f'stroke-linejoin="round">{_ICONS[icon]}</g>'
        )
        body.append(
            _text(cx, top + 92, title, size=19, fill=theme.text, weight=600, anchor="middle")
        )
        body.append(_text(cx, top + 117, subtitle, size=15, fill=theme.muted, anchor="middle"))
        if i < len(_STAGES) - 1:
            ax = x + node_w + gap / 2
            ay = top + node_h / 2
            body.append(
                f'<path d="M{_n(ax - 4)} {_n(ay - 6)} L{_n(ax + 3)} {_n(ay)} '
                f'L{_n(ax - 4)} {_n(ay + 6)}" fill="none" stroke="{theme.muted}" '
                'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
            )
    return _svg(
        width,
        height,
        "PassageWatch pipeline",
        "Sonar video, detection with YOLOX-Tiny on ONNX Runtime, Kalman tracking and counting "
        "by the start/end rule run automatically in one CPU worker; the technician then "
        "reviews a prioritized queue with random audits and exports CSV or JSON.",
        body,
    )


# --- Service -------------------------------------------------------------------------------


def _card_text(
    cx: float, top: float, title: str, lines: Sequence[str], theme: Theme, size: float = 17
) -> list[str]:
    out = [_text(cx, top, title, size=size, fill=theme.text, weight=600, anchor="middle")]
    for i, line in enumerate(lines):
        out.append(_text(cx, top + 22 + i * 19, line, size=13.5, fill=theme.muted, anchor="middle"))
    return out


def _card(
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: Sequence[str],
    theme: Theme,
    *,
    stroke: str | None = None,
    title_size: float = 17,
) -> list[str]:
    width = "1.5" if stroke else "1"
    return [
        f'<rect x="{_n(x)}" y="{_n(y)}" width="{_n(w)}" height="{_n(h)}" rx="12" '
        f'fill="{theme.card}" stroke="{stroke or theme.card_border}" stroke-width="{width}"/>',
        *_card_text(x + w / 2, y + h / 2 - 10 * len(lines) + 6, title, lines, theme, title_size),
    ]


def _cylinder(
    x: float, y: float, w: float, h: float, title: str, lines: Sequence[str], theme: Theme
) -> list[str]:
    ry = 9.0
    rx = w / 2
    body = (
        f"M{_n(x)} {_n(y + ry)} A{_n(rx)} {_n(ry)} 0 0 1 {_n(x + w)} {_n(y + ry)} "
        f"V{_n(y + h - ry)} A{_n(rx)} {_n(ry)} 0 0 1 {_n(x)} {_n(y + h - ry)} Z"
    )
    return [
        f'<path d="{body}" fill="{theme.card}" stroke="{theme.card_border}"/>',
        f'<ellipse cx="{_n(x + rx)}" cy="{_n(y + ry)}" rx="{_n(rx)}" ry="{_n(ry)}" '
        f'fill="{theme.neutral_wash}" stroke="{theme.card_border}"/>',
        *_card_text(x + rx, y + ry + h / 2 - 10 * len(lines) + 6, title, lines, theme),
    ]


def _connector(
    points: Sequence[Point], color: str, *, dashed: bool = False, both: bool = False
) -> str:
    path = "M" + " L".join(f"{_n(x)} {_n(y)}" for x, y in points)
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    parts = [
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.8" '
        f'stroke-linejoin="round"{dash}/>',
        _arrowhead(points[-2], _back_off(points[-2], points[-1], 5), 13, color),
    ]
    if both:
        parts.append(_arrowhead(points[1], _back_off(points[1], points[0], 5), 13, color))
    return "\n".join(parts)


def _back_off(start: Point, end: Point, distance: float) -> Point:
    """``end`` moved ``distance`` back towards ``start`` (so an arrow tip meets the edge)."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    return (end[0] - dx / length * distance, end[1] - dy / length * distance)


def service_svg(theme: Theme) -> str:
    """The deployment: browser, API, SQLite, worker, bundle, volume and Prometheus."""
    width, height = 1146, 400
    row_y, row_h = 170.0, 96.0
    mid = row_y + row_h / 2
    api = (230.0, 150.0)
    db = (450.0, 170.0)
    worker = (690.0, 180.0)
    bundle = (940.0, 170.0)
    prom = (450.0, 60.0, 170.0, 60.0)
    media = (562.0, 296.0, 190.0, 64.0)
    api_c, worker_c = api[0] + api[1] / 2, worker[0] + worker[1] / 2
    ink = theme.muted
    body: list[str] = [
        '<rect x="206" y="16" width="924" height="368" rx="18" fill="none" '
        f'stroke="{theme.card_border}" stroke-width="1.5" stroke-dasharray="7 6"/>',
        _text(228, 44, "One CPU host · Docker Compose", size=14, fill=ink, weight=600),
        # Requests and data.
        _connector([(166, mid), (api[0], mid)], theme.human),
        _connector([(api[0] + api[1], mid), (db[0], mid)], ink, both=True),
        _connector([(db[0] + db[1], mid), (worker[0], mid)], ink, both=True),
        _connector(
            [
                (api_c, row_y + row_h),
                (api_c, media[1] + media[3] / 2),
                (media[0], media[1] + media[3] / 2),
            ],
            ink,
        ),
        _connector(
            [
                (worker_c, row_y + row_h),
                (worker_c, media[1] + media[3] / 2),
                (media[0] + media[2], media[1] + media[3] / 2),
            ],
            ink,
        ),
        # Loads and scrapes.
        _connector([(bundle[0], mid), (worker[0] + worker[1], mid)], theme.accent, dashed=True),
        _text(
            (bundle[0] + worker[0] + worker[1]) / 2,
            mid - 10,
            "loads",
            size=12.5,
            fill=theme.accent,
            anchor="middle",
        ),
        _connector(
            [(prom[0], prom[1] + prom[3] / 2), (api_c, prom[1] + prom[3] / 2), (api_c, row_y)],
            ink,
            dashed=True,
        ),
        _connector(
            [
                (prom[0] + prom[2], prom[1] + prom[3] / 2),
                (worker_c, prom[1] + prom[3] / 2),
                (worker_c, row_y),
            ],
            ink,
            dashed=True,
        ),
        _text(
            (prom[0] + api_c) / 2,
            prom[1] + prom[3] / 2 - 8,
            "scrape",
            size=12.5,
            fill=ink,
            anchor="middle",
        ),
        _text(
            (prom[0] + prom[2] + worker_c) / 2,
            prom[1] + prom[3] / 2 - 8,
            "scrape",
            size=12.5,
            fill=ink,
            anchor="middle",
        ),
        # Components.
        *_card(
            16,
            row_y,
            150,
            row_h,
            "Review page",
            ["browser", "TypeScript"],
            theme,
            stroke=theme.human,
        ),
        *_card(
            api[0], row_y, api[1], row_h, "API", ["FastAPI + Pydantic", "202, then poll"], theme
        ),
        *_cylinder(
            db[0],
            row_y,
            db[1],
            row_h,
            "SQLite (WAL)",
            ["jobs · revisions", "append-only reviews"],
            theme,
        ),
        *_card(
            worker[0],
            row_y,
            worker[1],
            row_h,
            "Inference worker",
            ["one job at a time", "lease + heartbeat"],
            theme,
            stroke=theme.accent,
            title_size=16.5,
        ),
        *_card(
            bundle[0],
            row_y,
            bundle[1],
            row_h,
            "Release bundle",
            ["ONNX + configs", "hash-checked"],
            theme,
        ),
        *_card(*prom, "Prometheus", ["metrics · alerts"], theme),
        *_cylinder(*media, "Media + artifacts", ["tracks · audits"], theme),
    ]
    return _svg(
        width,
        height,
        "PassageWatch service",
        "On one CPU host under Docker Compose: the browser review page talks to a FastAPI "
        "API; the API and one inference worker share SQLite in WAL mode (jobs, revisions, "
        "append-only reviews) and a media and artifacts volume. The worker leases one job at "
        "a time with heartbeats and loads a hash-checked release bundle once. Prometheus "
        "scrapes the API and the worker.",
        body,
    )


# --- Counting rule -------------------------------------------------------------------------

_RULE_PANELS: tuple[tuple[str, str, bool, tuple[Point, ...]], ...] = (
    (
        "Starts left, ends right",
        "+1 → right",
        True,
        ((38, 104), (90, 98), (140, 86), (192, 74), (222, 70)),
    ),
    (
        "Starts right, ends left",
        "+1 ← left",
        True,
        ((218, 58), (168, 66), (116, 80), (66, 92), (34, 96)),
    ),
    (
        "Crosses, then comes back",
        "0, not 2",
        False,
        ((40, 116), (104, 108), (166, 92), (196, 70), (160, 50), (96, 42), (64, 40)),
    ),
    (
        "Approaches, never crosses",
        "0",
        False,
        ((30, 52), (64, 60), (94, 76), (98, 96), (72, 110), (40, 114)),
    ),
)


def counting_rule_svg(theme: Theme) -> str:
    """Four tracks and what each contributes under ``cfc-compatible-v1``."""
    width, height = 1140, 318
    margin, gap = 30.0, 24.0
    panel_w = (width - 2 * margin - 3 * gap) / 4
    frame_h = 150.0
    top = 12.0
    body: list[str] = []
    for i, (title, result, counted, points) in enumerate(_RULE_PANELS):
        x = margin + i * (panel_w + gap)
        mid = x + panel_w / 2
        body.append(f'<g transform="translate({_n(x)} {_n(top)})">')
        body.append(
            f'<rect width="{_n(panel_w)}" height="{_n(frame_h)}" rx="12" fill="{SONAR_BG}" '
            f'stroke="{theme.card_border}"/>'
        )
        # A faint sonar wedge behind the track.
        cx = panel_w / 2
        body.append(
            f'<path d="M{_n(cx - 52)} 8 L{_n(cx - 112)} {_n(frame_h - 8)} '
            f'Q{_n(cx)} {_n(frame_h + 8)} {_n(cx + 112)} {_n(frame_h - 8)} L{_n(cx + 52)} 8 Z" '
            f'fill="{SONAR_FAN}"/>'
        )
        body.append(
            f'<line x1="{_n(cx)}" y1="8" x2="{_n(cx)}" y2="{_n(frame_h - 8)}" '
            f'stroke="{SONAR_LINE}" stroke-opacity="0.75" stroke-width="2" '
            'stroke-dasharray="5 5"/>'
        )
        body.append(_text(14, 26, "left", size=13, fill="#8b949e", weight=600))
        body.append(
            _text(panel_w - 14, 26, "right", size=13, fill="#8b949e", weight=600, anchor="end")
        )
        scale = panel_w / 252
        scaled = [(px * scale, py) for px, py in points]
        body.append(_track(scaled, color=TRACK, background=SONAR_BG))
        body.append("</g>")
        body.append(
            _text(
                mid,
                top + frame_h + 36,
                title,
                size=18,
                fill=theme.text,
                weight=600,
                anchor="middle",
            )
        )
        size = 17.0
        pill_w = _text_width(result, size) + size * 1.4
        body.append(
            _pill(
                mid - pill_w / 2,
                top + frame_h + 72,
                result,
                size=size,
                fill=theme.accent_wash if counted else theme.neutral_wash,
                ink=theme.accent if counted else theme.muted,
            )
        )
    # Key.
    key_y = height - 18.0
    items: list[tuple[str, str]] = [
        ("start", "track starts"),
        ("end", "track ends"),
        ("line", "counting line"),
    ]
    total = sum(30 + _text_width(label, 14) + 26 for _, label in items)
    kx = width / 2 - total / 2
    for kind, label in items:
        if kind == "start":
            body.append(
                f'<circle cx="{_n(kx + 8)}" cy="{_n(key_y - 5)}" r="5.5" fill="none" '
                f'stroke="{theme.accent}" stroke-width="2.2"/>'
            )
        elif kind == "end":
            body.append(_arrowhead((kx, key_y - 5), (kx + 9, key_y - 5), 14, theme.accent))
        else:
            body.append(
                f'<line x1="{_n(kx + 8)}" y1="{_n(key_y - 14)}" x2="{_n(kx + 8)}" '
                f'y2="{_n(key_y + 4)}" stroke="{theme.muted}" stroke-width="2" '
                'stroke-dasharray="4 3"/>'
            )
        body.append(_text(kx + 24, key_y, label, size=14, fill=theme.muted))
        kx += 30 + _text_width(label, 14) + 26
    return _svg(
        width,
        height,
        "How a track is counted",
        "Policy cfc-compatible-v1 counts a completed track by the side of the counting line "
        "where it starts and where it ends. Starting left and ending right counts one passage "
        "to the right; starting right and ending left counts one to the left. A track that "
        "crosses and comes back counts zero, not two, and a track that never crosses counts "
        "zero.",
        body,
    )


# --- Test results --------------------------------------------------------------------------


def _bar(x0: float, y: float, length: float, thickness: float, fill: str) -> str:
    """A horizontal bar, square at the baseline with a 4px rounded data end."""
    r = min(4.0, thickness / 2, max(length, 0.0))
    if length <= 0:
        return ""
    return (
        f'<path d="M{_n(x0)} {_n(y)} H{_n(x0 + length - r)} '
        f"A{_n(r)} {_n(r)} 0 0 1 {_n(x0 + length)} {_n(y + r)} "
        f"V{_n(y + thickness - r)} "
        f"A{_n(r)} {_n(r)} 0 0 1 {_n(x0 + length - r)} {_n(y + thickness)} "
        f'H{_n(x0)} Z" fill="{fill}"/>'
    )


def test_results_svg(results: BenchmarkResults, theme: Theme) -> str:
    """Grouped bars: count errors per 100 true passages at each test location."""
    width = 1100
    label_x, plot_x0, plot_x1 = 30.0, 262.0, 1040.0
    domain = 56.0
    thickness, bar_gap, group_gap = 14.0, 3.0, 26.0
    header = 140.0
    groups: list[tuple[str, str, dict[str, float]]] = [
        (loc.name, f"{loc.clips} clips · {loc.passages:,} passages", loc.nmae)
        for loc in results.locations
    ]
    groups.append(("Macro average", "4 locations, equal weight", results.macro))
    group_h = len(SYSTEMS) * thickness + (len(SYSTEMS) - 1) * bar_gap
    separator = 18.0
    plot_top = header
    plot_bottom = plot_top + len(groups) * group_h + (len(groups) - 1) * group_gap + separator
    height = int(plot_bottom + 92)

    def sx(value: float) -> float:
        return plot_x0 + (plot_x1 - plot_x0) * value / domain

    body: list[str] = [
        _text(
            label_x,
            40,
            "Count errors per 100 true passages, official CFC test locations",
            size=24,
            fill=theme.text,
            weight=700,
        ),
        _text(
            label_x,
            70,
            f"Frozen release, evaluated once on {results.clips:,} clips "
            f"({results.passages:,} passages) at locations never used for training or "
            "setting selection. Lower is better.",
            size=16,
            fill=theme.muted,
        ),
    ]
    # Legend.
    column = (width - 2 * label_x) / len(SYSTEMS)
    for i, ((key, label), color) in enumerate(zip(SYSTEMS, theme.series, strict=True)):
        lx = label_x + i * column
        name = label.format(release=results.release.removeprefix("passagewatch-"))
        body.append(f'<rect x="{_n(lx)}" y="96" width="14" height="14" rx="3" fill="{color}"/>')
        weight = 600 if key == "release" else 400
        body.append(_text(lx + 22, 108, name, size=16, fill=theme.text, weight=weight))
    # Grid and axis.
    for tick in range(0, 60, 10):
        gx = sx(tick)
        stroke = theme.axis if tick == 0 else theme.grid
        body.append(
            f'<line x1="{_n(gx)}" y1="{_n(plot_top - 8)}" x2="{_n(gx)}" '
            f'y2="{_n(plot_bottom + 4)}" stroke="{stroke}" stroke-width="1"/>'
        )
        body.append(
            _text(
                gx,
                plot_bottom + 26,
                str(tick),
                size=14,
                fill=theme.muted,
                anchor="middle",
                extra=' font-variant-numeric="tabular-nums"',
            )
        )
    body.append(
        _text(
            (plot_x0 + plot_x1) / 2,
            plot_bottom + 52,
            "count errors per 100 true passages (directional nMAE × 100)",  # noqa: RUF001
            size=14,
            fill=theme.muted,
            anchor="middle",
        )
    )
    y = plot_top
    for g, (name, detail, values) in enumerate(groups):
        if g == len(groups) - 1:
            sep_y = y - group_gap / 2 + separator / 2 - 4
            body.append(
                f'<line x1="{_n(label_x)}" y1="{_n(sep_y)}" x2="{_n(plot_x1)}" '
                f'y2="{_n(sep_y)}" stroke="{theme.axis}" stroke-width="1"/>'
            )
            y += separator
        center = y + group_h / 2
        body.append(_text(label_x, center - 3, name, size=17, fill=theme.text, weight=600))
        body.append(_text(label_x, center + 18, detail, size=14, fill=theme.muted))
        for b, ((key, _), color) in enumerate(zip(SYSTEMS, theme.series, strict=True)):
            by = y + b * (thickness + bar_gap)
            value = values[key] * 100
            body.append(_bar(plot_x0, by, sx(value) - plot_x0, thickness, color))
            release = key == "release"
            body.append(
                _text(
                    sx(value) + 7,
                    by + thickness - 2,
                    f"{value:.1f}",
                    size=14,
                    fill=theme.text if release else theme.muted,
                    weight=700 if release else 400,
                    extra=' font-variant-numeric="tabular-nums"',
                )
            )
        y += group_h + group_gap
    return _svg(
        width,
        height,
        "Test results",
        "Count errors per 100 true passages at the four official CFC test locations and their "
        "macro average. "
        + " ".join(
            f"{name}: "
            + ", ".join(
                f"{label.format(release=results.release.removeprefix('passagewatch-'))} "
                f"{values[key] * 100:.1f}"
                for key, label in SYSTEMS
            )
            + "."
            for name, _, values in groups
        ),
        body,
    )


# --- All figures ---------------------------------------------------------------------------


def all_figures(results: BenchmarkResults) -> dict[str, str]:
    """File name -> SVG text for every README figure."""
    figures = {"banner.svg": banner_svg()}
    for theme in THEMES:
        figures[f"pipeline-{theme.name}.svg"] = pipeline_svg(theme)
        figures[f"service-{theme.name}.svg"] = service_svg(theme)
        figures[f"counting-rule-{theme.name}.svg"] = counting_rule_svg(theme)
        figures[f"test-results-{theme.name}.svg"] = test_results_svg(results, theme)
    return figures
