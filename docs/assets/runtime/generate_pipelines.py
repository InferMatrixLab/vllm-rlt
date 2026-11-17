"""Original vLLM-RLT schematic. Pillow only; no remote assets.
Reproduce: python generate_pipelines.py --output-dir .
Source: v0.2.0 c08b9d79834bb8b4ff283b15bae2c14c559f6b52;
docs/unified_execution.md (Inside one stage), docs/runtime.md (WCPB),
docs/async-speculative.md (Pipelined self-speculation).
"""

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 600
WHITE = "#FFFFFF"
NAVY = "#14263D"
BLUE = "#2878D0"
YELLOW = "#F5CF56"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(n, b=False):
    return ImageFont.truetype(BOLD if b else FONT, n)


def txt(d, xy, s, n=18, fill=NAVY, b=False):
    d.text(xy, s, font=font(n, b), fill=fill)


def center(d, xy, s, n=18, fill=NAVY, b=False):
    box = d.textbbox((0, 0), s, font=font(n, b))
    txt(
        d,
        (xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2 - 4),
        s,
        n,
        fill,
        b,
    )


def arrow(d, p1, p2, col=BLUE):
    d.line((p1, p2), fill=col, width=3)
    a = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    r = 9
    d.polygon(
        [
            p2,
            (p2[0] - r * math.cos(a - 0.5), p2[1] - r * math.sin(a - 0.5)),
            (p2[0] - r * math.cos(a + 0.5), p2[1] - r * math.sin(a + 0.5)),
        ],
        fill=col,
    )


def frame(t):
    im = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(im)
    phase = int(t // 10)
    local = t % 10
    step = min(7, int(max(0, local - 1) / 1.1))
    txt(d, (35, 20), "vLLM-RLT", 28, b=True)
    txt(d, (235, 25), "Loop-level inference, made visible", 24)
    d.line((35, 67, 1165, 67), fill=BLUE, width=3)
    d.rounded_rectangle((35, 90, 1165, 515), radius=20, outline=BLUE, width=2)
    txt(d, (55, 103), "STAGE", 16, BLUE, True)
    txt(d, (135, 103), "deployment + resource boundary", 16)
    # ownership, no implied scheduler direct runner call
    d.rounded_rectangle((55, 145, 292, 468), radius=14, outline=NAVY, width=2)
    center(d, (173, 168), "StageEngine", 21, b=True)
    for yy, label, sub in [
        (197, "Scheduler", "select + apply progress"),
        (302, "Executor", "submit selected work"),
    ]:
        d.rounded_rectangle((72, yy, 275, yy + 66), radius=9, outline=BLUE, width=2)
        center(d, (173, yy + 22), label, 20, b=True)
        center(d, (173, yy + 47), sub, 13)
    arrow(d, (173, 266), (173, 298))
    txt(d, (184, 275), "via engine", 11)
    center(d, (173, 396), "Shared model runtime", 15, b=True)
    center(d, (173, 426), "hidden state + KV", 16)
    center(d, (173, 450), "retained across steps", 13)
    arrow(d, (294, 326), (326, 326))
    title = "01  WCPB · prefill wavefront" if phase == 0 else "02  Decode · token pipeline"
    txt(d, (334, 142), title, 25, b=True)
    txt(
        d,
        (334, 177),
        "Chunks regroup at each recurrent depth"
        if phase == 0
        else "Experimental self-speculation · accepted path",
        16,
        BLUE,
    )
    # grid 4 rows, 4 columns + output
    gx, gy = 425, 226
    cw, rh = 146, 49
    for k in range(4):
        center(d, (gx + k * cw + cw / 2, 210), f"Loop {k + 1}", 16, b=True)
    for r in range(4):
        center(
            d,
            (372, gy + r * rh + rh / 2),
            f"Chunk {r}" if phase == 0 else f"Input x{r}",
            15,
        )
        for c in range(4):
            x = gx + c * cw
            y = gy + r * rh
            d.rounded_rectangle(
                (x + 3, y + 3, x + cw - 5, y + rh - 5), radius=6, outline=BLUE, width=1
            )
            pos = step - r
            if c < pos:
                d.line((x + 56, y + 24, x + 66, y + 32, x + 87, y + 15), fill=BLUE, width=2)
            elif c == pos and pos >= 0:
                d.rounded_rectangle((x + 5, y + 5, x + cw - 7, y + rh - 7), radius=5, fill=YELLOW)
                center(
                    d,
                    (x + cw / 2, y + rh / 2),
                    f"C{r} : R{c + 1}" if phase == 0 else f"x{r} : R{c + 1}",
                    17,
                    b=True,
                )
        if step - r >= 4:
            center(
                d,
                (1062, gy + r * rh + rh / 2),
                "done" if phase == 0 else f"→ x{r + 1}",
                15,
                BLUE,
                True,
            )
    txt(d, (337, 438), "Each row keeps its own depth and state.", 16, b=True)
    txt(
        d,
        (337, 464),
        "Full prompt depth before first output."
        if phase == 0
        else "Rejection discards dependent tokens; state resumes at the valid frontier.",
        13,
    )
    # scene timeline only not measured time
    txt(d, (55, 489), "ONE RUNTIME • TWO DISTINCT EXECUTION EXAMPLES", 12, BLUE, True)
    txt(
        d,
        (35, 535),
        "Schematic, not a timing or performance measurement. Four loops shown for illustration.",
        14,
    )
    txt(
        d,
        (35, 560),
        "v0.2.0 · WCPB: opt-in, LAST_EXITED KV. "
        "Decode pipeline: greedy + LAST_EXITED; block remains default.",
        13,
    )
    for i in range(2):
        d.rounded_rectangle(
            (1040 + i * 65, 35, 1090 + i * 65, 42),
            radius=3,
            fill=BLUE if i == phase else YELLOW,
        )
    return im


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = [frame(i / 10) for i in range(200)]
    frames[43].save(out / "rlt-loop-pipelines.png")
    frames[143].save(out / "rlt-loop-pipelines-decode-qa.png")
    frames[0].save(
        out / "rlt-loop-pipelines.gif",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        optimize=True,
        disposal=1,
    )
    print(out / "rlt-loop-pipelines.gif")


if __name__ == "__main__":
    main()
