"""
The voice-over script as a PDF to read from while recording.

    python movie/make_script_pdf.py      # -> movie/vault_movie_script.pdf

The words come from MOVIE_SCRIPT.md; the start time of each scene comes from
the narration clips make_movie.py timed the movie with (movie/audio/), so run
make_movie.py first.
"""

import os
import sys
import wave

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_movie as M      # noqa: E402  (script reading and timing constants)

OUT = os.path.join(HERE, "vault_movie_script.pdf")
TITLE_S, END_S = 3.5, 4.0     # same as make_movie.py

# What the generated movie shows in each scene (so you know where you are).
ON_SCREEN = {
    1: "The safe: a hand presses the button, the LEDs go white, it shows 1, 4, 2 and the safe opens.",
    2: "A photo of my hand gets a 64 × 64 grid; a magnifier shows the red, green and blue numbers of 16 pixels.",
    3: "Three inputs times their weights add up in a neuron (1.01), then the ReLU graph: -1.3 becomes 0, 1.01 goes through.",
    4: "A 3 × 3 filter slides over a 10 × 10 picture and builds an edge map; real filters from my network; 448 vs ≈ 200,000.",
    5: "16 real feature maps fan out, then stacks of 16, 32, 64 and 128 maps.",
    6: "A blue 2 × 2 frame keeps the biggest number of each block; maps shrink 64 → 32 → 16 → 8; a moved finger, same answer.",
    7: "Blocks conv 1 to conv 4 stack up while circles on the photo grow (3, 8, 18, 38 pixels); then the bar chart of depths.",
    8: "128 feature maps squash into beads that pour into cups 0 to 5; cup 3 fills: 100%. Stamp: 98,694 weights.",
    9: "Picture → conv blocks → cups; random weights guess wrong; a red backpropagation arrow sweeps back and turns the dials.",
    10: "A marble rolls down the loss hill one step at a time, overshoots a bump, and settles in the valley.",
    11: "Photos of my hand pile up; one gets turned, zoomed, mirrored, darkened; some go in the TEST box; 95% right.",
    12: "Grad-CAM: the heat map on my watch (is 0, says 1, ✗), then three heat maps on fingers (✓).",
    13: "The conv blocks hop from the laptop onto the UNO Q, then the safe opens again: 1, 4, 2.",
}


def font_setup():
    base = "/System/Library/Fonts/Supplemental/"
    pdfmetrics.registerFont(TTFont("Body", base + "Georgia.ttf"))
    pdfmetrics.registerFont(TTFont("BodyBold", base + "Georgia Bold.ttf"))
    pdfmetrics.registerFont(TTFont("Sans", base + "Arial.ttf"))
    pdfmetrics.registerFont(TTFont("SansBold", base + "Arial Bold.ttf"))
    pdfmetrics.registerFont(TTFont("SansItalic", base + "Arial Italic.ttf"))
    # Arial Unicode for the arrows and symbols
    pdfmetrics.registerFont(TTFont("Sym", base + "Arial Unicode.ttf"))
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="SansBold", italic="SansItalic", boldItalic="SansBold")
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="Body", boldItalic="BodyBold")


def sym(text):
    """Wrap characters Arial lacks in the symbol font."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return "".join(f'<font name="Sym">{ch}</font>' if ch in "→✓✗≈←↓" else ch for ch in text)


def clock(t):
    m, s = divmod(int(round(t)), 60)
    return f"{m}:{s:02d}"


def scene_times(script):
    """(start, length) of each scene, the same arithmetic make_movie.py uses."""
    t, out = TITLE_S, []
    for n, _, _ in script:
        with wave.open(os.path.join(M.AUDIO_DIR, f"scene{n:02d}.wav")) as w:
            d = w.getnframes() / w.getframerate()
        length = round((M.LEAD_S + d + M.TAIL_S) * M.FPS) / M.FPS
        out.append((t, length))
        t += length
    return out, t + END_S


def main():
    font_setup()
    script = M.read_script()
    times, total = scene_times(script)

    INK = colors.HexColor("#2d2826")
    MUTED = colors.HexColor("#6b625b")
    ACCENT = colors.HexColor("#b5651d")
    BAND = colors.HexColor("#f4ead2")

    st = {
        "title": ParagraphStyle("title", fontName="SansBold", fontSize=24, leading=29, textColor=INK, alignment=TA_CENTER),
        "sub": ParagraphStyle("sub", fontName="Sans", fontSize=12, leading=16, textColor=MUTED, alignment=TA_CENTER),
        "h": ParagraphStyle("h", fontName="SansBold", fontSize=12, leading=16, textColor=INK, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("body", fontName="Sans", fontSize=11, leading=16, textColor=INK, spaceAfter=4),
        "small": ParagraphStyle("small", fontName="Sans", fontSize=10, leading=14, textColor=MUTED),
        "scene": ParagraphStyle("scene", fontName="SansBold", fontSize=15, leading=19, textColor=INK),
        "time": ParagraphStyle("time", fontName="SansBold", fontSize=12, leading=19, textColor=ACCENT, alignment=2),
        "screen": ParagraphStyle("screen", fontName="SansItalic", fontSize=10, leading=14, textColor=MUTED,
                                 spaceBefore=4, spaceAfter=8),
        "vo": ParagraphStyle("vo", fontName="Body", fontSize=15, leading=24, textColor=INK, spaceAfter=7),
        "toc": ParagraphStyle("toc", fontName="Sans", fontSize=11, leading=15, textColor=INK),
    }

    story = [Paragraph("How my finger lock's neural network works", st["title"]), Spacer(1, 6),
             Paragraph(f"Voice-over script · ME193 · Fiona Gettelman · movie length {clock(total)}", st["sub"]),
             Spacer(1, 18),
             Paragraph("How to record", st["h"])]
    for line in [
        "<b>1.</b> Watch <b>vault_movie_guide.mp4</b> once. A computer voice reads this script at the right pace, with captions.",
        "<b>2.</b> Open <b>vault_movie.mp4</b> (silent) in iMovie, choose <i>Record Voiceover</i>, and read the script while it plays. "
        "Each scene below shows the time it starts in the movie.",
        "<b>3.</b> Don't rush. Every scene has a short pause at the start and end. If you're a little slow or fast, "
        "just start the next scene when its start time comes up.",
        "<b>4.</b> Export, and the movie is done.",
    ]:
        story.append(Paragraph(line, st["body"]))
    story += [Spacer(1, 10), Paragraph("Scenes", st["h"])]

    rows = [[Paragraph("<b>Starts</b>", st["toc"]), Paragraph("<b>Scene</b>", st["toc"]), Paragraph("<b>Length</b>", st["toc"])],
            [Paragraph("0:00", st["toc"]), Paragraph("Title (no voice-over)", st["toc"]), Paragraph(f"{TITLE_S:.0f} s", st["toc"])]]
    for (n, title, _), (t, length) in zip(script, times):
        rows.append([Paragraph(clock(t), st["toc"]), Paragraph(f"{n}. {title}", st["toc"]),
                     Paragraph(f"{length:.0f} s", st["toc"])])
    rows.append([Paragraph(clock(total - END_S), st["toc"]), Paragraph("End card (no voice-over)", st["toc"]),
                 Paragraph(f"{END_S:.0f} s", st["toc"])])
    toc = Table(rows, colWidths=[0.8 * inch, 4.6 * inch, 0.9 * inch])
    toc.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
                             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
                             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story += [toc, PageBreak()]

    for (n, title, vo), (t, length) in zip(script, times):
        head = Table([[Paragraph(f"{n}. {sym(title)}", st["scene"]),
                       Paragraph(f"{clock(t)} – {clock(t + length)}", st["time"])]],
                     colWidths=[4.9 * inch, 1.6 * inch])
        head.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BAND),
                                  ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                                  ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                                  ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        block = [head, Paragraph("On screen: " + sym(ON_SCREEN.get(n, "")), st["screen"])]
        block += [Paragraph(sym(s), st["vo"]) for s in M.sentences(vo)]
        block.append(Spacer(1, 14))
        story.append(KeepTogether(block))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Sans", 9)
        canvas.setFillColor(MUTED)
        canvas.drawString(0.9 * inch, 0.55 * inch, "Vault Lock · voice-over script")
        canvas.drawRightString(letter[0] - 0.9 * inch, 0.55 * inch, f"page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                            topMargin=0.8 * inch, bottomMargin=0.9 * inch,
                            title="Vault Lock: voice-over script", author="Fiona Gettelman")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Saved {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
