# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""Render report blocks (see report.py) as an A4 PDF with reportlab."""
from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

INK = colors.HexColor("#1b1b1b")
GREY = colors.HexColor("#5f6368")
LINE = colors.HexColor("#c9ccd1")
HEAD = colors.HexColor("#eef0f3")
BOX = colors.HexColor("#e8f0fb")
RED = colors.HexColor("#b3261e")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=19, leading=23, alignment=TA_LEFT,
                            textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("s", parent=ss["Normal"], fontName="Helvetica", fontSize=11, leading=14, textColor=GREY, spaceAfter=8),
    "flag": ParagraphStyle("f", parent=ss["Normal"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=RED, spaceAfter=8),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=INK,
                         spaceBefore=10, spaceAfter=5),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=INK,
                         spaceBefore=7, spaceAfter=3),
    "p": ParagraphStyle("p", parent=ss["Normal"], fontName="Helvetica", fontSize=9.5, leading=13, textColor=INK, spaceAfter=5),
    "small": ParagraphStyle("sm", parent=ss["Normal"], fontName="Helvetica", fontSize=7.8, leading=10, textColor=INK),
    "smallb": ParagraphStyle("smb", parent=ss["Normal"], fontName="Helvetica-Bold", fontSize=7.8, leading=10, textColor=INK),
    "cap": ParagraphStyle("c", parent=ss["Normal"], fontName="Helvetica", fontSize=8, leading=10, textColor=GREY, spaceAfter=6),
    "code": ParagraphStyle("code", parent=ss["Code"], fontName="Courier", fontSize=7.6, leading=9.6, textColor=INK,
                           backColor=colors.HexColor("#f5f5f5"), borderPadding=5, spaceBefore=3, spaceAfter=8),
    "cover": ParagraphStyle("cv", parent=ss["Normal"], fontName="Helvetica", fontSize=8.6, leading=11.5, textColor=INK, spaceAfter=4),
}


def _inline(t: str) -> str:
    t = escape(str(t))
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"(?<![\w*])\*(?!\s)([^*]+?)\*(?![\w*])", r"<i>\1</i>", t)


def _table(header, rows, width):
    data = [[Paragraph(_inline(h), S["smallb"]) for h in header]]
    data += [[Paragraph(_inline(x), S["small"]) for x in r] for r in rows]
    n = len(header)
    # width: every column fits its longest word; long text columns share what is left
    def longest_word(i):
        return max(len(w) for x in [header[i]] + [r[i] for r in rows] for w in (str(x).split() or [""]))

    def longest_cell(i):
        return max(len(str(x)) for x in [header[i]] + [r[i] for r in rows])

    need = [max(longest_word(i) + 1, 3) for i in range(n)]
    want = [min(max(longest_cell(i), need[i]), 48) for i in range(n)]
    tot = sum(want)
    cw = [width * w / tot for w in want]
    # make sure no column is narrower than its longest word (about 0.52 em per char at 7.8 pt)
    char = 7.8 * 0.62
    short = [i for i in range(n) if cw[i] < need[i] * char + 6]
    for i in short:
        cw[i] = need[i] * char + 6
    rest = [i for i in range(n) if i not in short]
    if rest:
        spare = width - sum(cw[i] for i in short)
        tw = sum(want[i] for i in rest)
        for i in rest:
            cw[i] = max(spare * want[i] / tw, need[i] * char + 6)
    t = Table(data, colWidths=cw, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    return t


def _box(title, text, width):
    p = Paragraph(f"<b>{_inline(title)}.</b> {_inline(text)}", S["p"])
    t = Table([[p]], colWidths=[width])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BOX), ("LEFTPADDING", (0, 0), (-1, -1), 6),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t


def render(blocks: list, path: Path, footer_title: str, author_name="KEYUR SHAILESHKUMAR PATEL",
           orcid="0009-0008-5261-043X") -> Path:
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title=footer_title, author=author_name)
    W = doc.width
    F = []
    for b in blocks:
        k = b[0]
        if k == "cover":
            c = b[1]
            F.append(Paragraph(_inline(c["title"]), S["title"]))
            sub = c.get("subtitle", "")
            F.append(Paragraph(_inline(sub), S["flag"] if "SYNTHETIC" in sub or "NOT" in sub else S["sub"]))
            au = c.get("author") or {}
            if au:
                F.append(Paragraph(f"<b>{escape(au['name'])}</b>", S["cover"]))
                F.append(Paragraph(f"ORCID: {escape(au['orcid'])}", S["cover"]))
                for a in au.get("affiliations", []):
                    F.append(Paragraph(escape(a), S["cover"]))
                F.append(Paragraph(f"<i>{escape(au.get('note', ''))}</i>", S["cover"]))
                F.append(Spacer(1, 4))
            for line in c.get("lines", []):
                if line.startswith("Use of AI tools."):
                    F.append(Paragraph("<b>Use of AI tools.</b>" + _inline(line[len("Use of AI tools."):]), S["cover"]))
                else:
                    F.append(Paragraph(_inline(line), S["cover"]))
            F.append(Spacer(1, 6))
        elif k == "h1":
            F.append(Paragraph(_inline(b[1]), S["h1"]))
        elif k == "h2":
            F.append(Paragraph(_inline(b[1]), S["h2"]))
        elif k == "p":
            F.append(Paragraph(_inline(b[1]), S["p"]))
        elif k == "bullets":
            F.append(ListFlowable([ListItem(Paragraph(_inline(x), S["p"]), leftIndent=10) for x in b[1]],
                                  bulletType="bullet", start="•", leftIndent=10, bulletFontSize=8))
        elif k == "box":
            F += [Spacer(1, 2), _box(b[1], b[2], W), Spacer(1, 6)]
        elif k == "table":
            if b[2]:
                F += [_table(b[1], b[2], W), Spacer(1, 6)]
        elif k == "img":
            p = Path(b[1])
            if p.exists():
                from reportlab.lib.utils import ImageReader

                iw, ih = ImageReader(str(p)).getSize()
                w = W * (b[3] if len(b) > 3 else 0.85)
                F.append(KeepTogether([Image(str(p), width=w, height=w * ih / iw), Paragraph(_inline(b[2]), S["cap"])]))
        elif k == "code":
            F.append(Preformatted(b[1].rstrip(), S["code"], maxLineLength=110))
        elif k == "pagebreak":
            F.append(PageBreak())

    def foot(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GREY)
        canvas.drawCentredString(A4[0] / 2, 9 * mm, f"{footer_title} - {author_name} - ORCID {orcid} - page {d.page}")
        canvas.restoreState()

    doc.build(F, onFirstPage=foot, onLaterPages=foot)
    return path


def run_report_pdf(run_dir: Path, ctx: dict) -> Path:
    from .report import SHORT_TITLE, build_blocks

    B = ctx.get("_blocks") or build_blocks(run_dir, ctx)
    title = SHORT_TITLE + (" (SYNTHETIC)" if ctx.get("synthetic") else "")
    return render(B, run_dir / "REPORT.pdf", title)


def md_to_blocks(text: str) -> list:
    """Small Markdown reader for this project's own documents (headings, paragraphs, bullets, tables, code)."""
    B, para, lines = [], [], text.splitlines()

    def flush():
        if para:
            B.append(("p", " ".join(para)))
            para.clear()

    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if ln.startswith("```"):
            flush()
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            B.append(("code", "\n".join(lines[i + 1:j])))
            i = j + 1
            continue
        if ln.startswith("# "):
            flush()
            B.append(("title", ln[2:]))
        elif ln.startswith("## "):
            flush()
            B.append(("h1", ln[3:]))
        elif ln.startswith("### "):
            flush()
            B.append(("h2", ln[4:]))
        elif ln.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):
                    rows.append([c.replace("`", "") for c in cells])
                i += 1
            B.append(("table", rows[0], rows[1:]))
            continue
        elif ln.lstrip().startswith(("- ", "* ")) or re.match(r"^\d+\. ", ln.lstrip()):
            flush()
            items = []
            while i < len(lines) and (lines[i].lstrip().startswith(("- ", "* ")) or re.match(r"^\d+\. ", lines[i].lstrip())):
                items.append(re.sub(r"^(\s*[-*] |\s*\d+\. )", "", lines[i]).replace("`", ""))
                i += 1
            B.append(("bullets", items))
            continue
        elif ln.strip() in ("", "---"):
            flush()
        else:
            para.append(ln.strip().replace("`", ""))
        i += 1
    flush()
    return B


def md_file_to_pdf(md_path: Path, pdf_path: Path, footer_title: str) -> Path:
    B = md_to_blocks(Path(md_path).read_text())
    out = []
    for b in B:
        if b[0] == "title":
            out.append(("cover", {"title": b[1], "subtitle": "", "author": None, "lines": []}))
        else:
            out.append(b)
    return render(out, pdf_path, footer_title)
