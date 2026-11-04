"""Render report.md and its local figures into the distributable PDF."""
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
from urllib.parse import urljoin

from markdown_it import MarkdownIt
import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    Preformatted, KeepTogether, PageBreak,
)
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/MortiferumUrsus/llm-effective-context-research/blob/main/"
NAVY = colors.HexColor("#213d53")
TEAL = colors.HexColor("#287d8e")


def register_fonts():
    """Embed bundled fonts instead of relying on the PDF viewer's substitutes."""
    font_dir = Path(reportlab.__file__).resolve().parent / "fonts"
    for name, filename in (
        ("ReportSans", "Vera.ttf"),
        ("ReportSans-Bold", "VeraBd.ttf"),
        ("ReportSans-Italic", "VeraIt.ttf"),
        ("ReportSans-BoldItalic", "VeraBI.ttf"),
    ):
        pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    pdfmetrics.registerFontFamily(
        "ReportSans", normal="ReportSans", bold="ReportSans-Bold",
        italic="ReportSans-Italic", boldItalic="ReportSans-BoldItalic",
    )


def clean(text):
    return text.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")


def inline(children):
    result = []
    for token in children or []:
        if token.type == "text":
            result.append(escape(clean(token.content)))
        elif token.type == "code_inline":
            result.append('<font name="ReportSans" size="8.5">'+escape(clean(token.content))+"</font>")
        elif token.type == "strong_open": result.append("<b>")
        elif token.type == "strong_close": result.append("</b>")
        elif token.type == "em_open": result.append("<i>")
        elif token.type == "em_close": result.append("</i>")
        elif token.type == "link_open":
            href = token.attrGet("href")
            target = href if href.startswith(("http://", "https://")) else urljoin(REPO, href)
            result.append('<a href="'+escape(target, quote=True)+'" color="#287d8e">')
        elif token.type == "link_close": result.append("</a>")
        elif token.type in ("softbreak", "hardbreak"): result.append(" ")
    return "".join(result)


def build(source: Path, destination: Path):
    register_fonts()
    styles = getSampleStyleSheet()
    body = ParagraphStyle("ResearchBody", fontName="ReportSans", fontSize=9.5, leading=13.0,
        textColor=NAVY, spaceAfter=7, allowWidows=0, allowOrphans=0)
    caption = ParagraphStyle("ResearchCaption", parent=body, fontSize=8, leading=10.5,
        textColor=colors.HexColor("#536572"), spaceBefore=5, spaceAfter=12)
    heading = {
        "h1": ParagraphStyle("ReportTitle", parent=body, fontName="ReportSans-Bold", fontSize=27,
                             leading=31, spaceBefore=4, spaceAfter=12, keepWithNext=True),
        "h2": ParagraphStyle("Section", parent=body, fontName="ReportSans-Bold", fontSize=13.5,
                             leading=17, spaceBefore=15, spaceAfter=8, keepWithNext=True),
        "h3": ParagraphStyle("Subsection", parent=body, fontName="ReportSans-Bold", fontSize=10.5,
                             leading=14, spaceBefore=10, spaceAfter=5, keepWithNext=True),
    }
    bullet = ParagraphStyle("ResearchBullet", parent=body, leftIndent=12, firstLineIndent=0,
                            bulletIndent=1, spaceAfter=5)
    tablebody = ParagraphStyle("TableBody", parent=body, fontSize=8.1, leading=10.8, spaceAfter=0)
    tablehead = ParagraphStyle("TableHead", parent=tablebody, fontName="ReportSans-Bold", textColor=colors.white)
    code = ParagraphStyle("ResearchCode", fontName="ReportSans", fontSize=8.4, leading=11.5,
                          textColor=NAVY, backColor=colors.HexColor("#f1f5f7"), borderPadding=9,
                          spaceBefore=7, spaceAfter=11)
    destination.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
        topMargin=18*mm, bottomMargin=18*mm, title="Effective Context in LLM Systems",
        author="Mikhail Sadovskiy", subject="An empirical study across tasks and processing methods",
        creator="Effective Context report renderer")
    tokens = MarkdownIt("commonmark").enable("table").parse(source.read_text(encoding="utf-8"))
    story = []
    i, list_depth = 0, 0
    pending_image = None
    while i < len(tokens):
        token = tokens[i]
        if token.type == "heading_open":
            if pending_image:
                story.append(pending_image); pending_image = None
            if tokens[i+1].content.startswith(("2. Defining", "3. Experimental", "Appendix B.")):
                story.append(PageBreak())
            story.append(Paragraph(inline(tokens[i+1].children), heading.get(token.tag, heading["h3"])))
            i += 3
            continue
        if token.type in ("bullet_list_open", "ordered_list_open"):
            list_depth += 1
        elif token.type in ("bullet_list_close", "ordered_list_close"):
            list_depth -= 1
        elif token.type == "paragraph_open":
            content = tokens[i+1]
            images = [t for t in content.children or [] if t.type == "image"]
            if images:
                path = (source.parent/images[0].attrGet("src")).resolve()
                if not path.is_relative_to(source.parent.resolve()):
                    raise ValueError("Figure must be inside the report directory")
                with PILImage.open(path) as pic:
                    width, height = pic.size
                scale = min(doc.width/width, 310/height)
                pending_image = Image(str(path), width*scale, height*scale)
            else:
                text = inline(content.children)
                is_caption = content.content.startswith("*Figure ")
                paragraph = Paragraph(text, caption if is_caption else (bullet if list_depth else body),
                                      bulletText="-" if list_depth else None)
                if pending_image:
                    story.append(KeepTogether([pending_image, paragraph]))
                    pending_image = None
                else:
                    story.append(paragraph)
            i += 3
            continue
        elif token.type == "fence":
            story.append(Preformatted(clean(token.content.rstrip()), code))
        elif token.type == "table_open":
            rows, current, plain = [], [], []
            j = i+1
            while tokens[j].type != "table_close":
                t = tokens[j]
                if t.type == "tr_open":
                    current, plain_row = [], []
                elif t.type == "inline":
                    current.append(inline(t.children)); plain_row.append(t.content)
                elif t.type == "tr_close":
                    rows.append(current); plain.append(plain_row)
                j += 1
            ncols = len(rows[0])
            weights = [max(12, min(48, max(len(row[c]) for row in plain))) for c in range(ncols)]
            weights[0] = max(16, weights[0])
            widths = [doc.width*w/sum(weights) for w in weights]
            values = [[Paragraph(cell, tablehead if n == 0 else tablebody) for cell in row]
                      for n, row in enumerate(rows)]
            table = Table(values, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f3f6f8"), colors.white]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#ccd6dc")),
            ]))
            story.append(KeepTogether([Spacer(1, 4), table, Spacer(1, 11)]))
            i = j+1
            continue
        i += 1
    if pending_image:
        story.append(pending_image)

    def page(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#ccd6dc"))
        canvas.line(doc.leftMargin, 14*mm, A4[0]-doc.rightMargin, 14*mm)
        canvas.setFont("ReportSans", 7.5)
        canvas.setFillColor(colors.HexColor("#657984"))
        canvas.drawString(doc.leftMargin, 10*mm, "Mikhail Sadovskiy | Effective Context in LLM Systems")
        canvas.drawRightString(A4[0]-doc.rightMargin, 10*mm, str(document.page))
        canvas.restoreState()
    doc.build(story, onFirstPage=page, onLaterPages=page)
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"report.md")
    parser.add_argument("--out", type=Path, default=ROOT/"output/pdf/effective_context_report.pdf")
    args = parser.parse_args()
    build(args.source.resolve(), args.out.resolve())
