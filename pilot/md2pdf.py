# Pilot-side helper: Markdown -> HTML -> PDF (Chromium print). Usage: python md2pdf.py in.md out.pdf
import sys, subprocess, markdown, pathlib, tempfile
src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
body = markdown.markdown(src.read_text(), extensions=["tables", "fenced_code", "sane_lists"])
css = """body{font-family:Georgia,'DejaVu Serif',serif;font-size:10.5pt;line-height:1.45;color:#111;margin:0}
h1{font-size:19pt;margin:0 0 6pt}h2{font-size:14pt;margin:18pt 0 6pt;border-bottom:1px solid #999;padding-bottom:2pt}h3{font-size:11.5pt;margin:12pt 0 4pt}
table{border-collapse:collapse;width:100%;margin:6pt 0 10pt;font-size:8.6pt;font-family:'DejaVu Sans',Arial,sans-serif}
th,td{border:1px solid #888;padding:3pt 5pt;text-align:left;vertical-align:top}th{background:#eee}
tr{page-break-inside:avoid}blockquote{border-left:3px solid #555;margin:8pt 0;padding:4pt 10pt;background:#f4f4f4}
pre{background:#f4f4f4;padding:6pt;font-size:8.3pt;white-space:pre-wrap;border:1px solid #ccc}code{font-size:90%}a{color:#111}"""
html = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{src.stem}</title><style>{css}</style></head><body>{body}</body></html>"
h = out.with_suffix(".html"); h.write_text(html)
js = f"""const {{chromium}}=require('playwright');(async()=>{{const b=await chromium.launch();const p=await b.newPage();
await p.goto('file://{h.resolve()}');await p.pdf({{path:'{out.resolve()}',format:'A4',margin:{{top:'16mm',bottom:'16mm',left:'15mm',right:'15mm'}},displayHeaderFooter:true,
headerTemplate:'<span></span>',footerTemplate:'<div style="font-size:7pt;width:100%;text-align:center;color:#555">{src.stem} · exploratory · page <span class="pageNumber"></span> of <span class="totalPages"></span></div>'}});await b.close();}})();"""
subprocess.run(["node", "-e", js], check=True)
print("pdf", out, out.stat().st_size)
