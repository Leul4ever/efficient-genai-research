import subprocess
import sys
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parent.parent
paper_md_path = ROOT / "paper" / "paper.md"
html_out_path = ROOT / "paper" / "paper.html"
pdf_out_path = ROOT / "paper" / "paper.pdf"

md_content = paper_md_path.read_text(encoding="utf-8")

# Fix image paths for HTML: convert ../results/figures to file:///...
figures_dir = (ROOT / "results" / "figures").as_uri()
md_content_html = md_content.replace("../results/figures", figures_dir)

# Convert Markdown to HTML
html_body = markdown.markdown(
    md_content_html,
    extensions=["tables", "fenced_code", "toc", "attr_list"]
)

# Academic CSS Styling
css = """
@page {
    size: A4;
    margin: 20mm 18mm 20mm 18mm;
    @bottom-center {
        content: counter(page);
        font-family: 'Times New Roman', Times, serif;
        font-size: 10pt;
    }
}

body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #111;
    background-color: #fff;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}

h1 {
    font-size: 22pt;
    font-weight: bold;
    text-align: center;
    margin-top: 20pt;
    margin-bottom: 8pt;
    line-height: 1.2;
}

h2 {
    font-size: 14pt;
    font-weight: bold;
    border-bottom: 1px solid #ccc;
    padding-bottom: 4px;
    margin-top: 18pt;
    margin-bottom: 8pt;
}

h3 {
    font-size: 12pt;
    font-weight: bold;
    margin-top: 14pt;
    margin-bottom: 6pt;
}

p {
    text-align: justify;
    margin-bottom: 10pt;
    text-indent: 0;
}

em {
    font-style: italic;
}

strong {
    font-weight: bold;
}

blockquote {
    margin: 12pt 20pt;
    padding: 8pt 12pt;
    background-color: #f8f9fa;
    border-left: 4px solid #0056b3;
    font-style: italic;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 14pt 0;
    font-size: 9.5pt;
    font-family: 'Arial', sans-serif;
}

th, td {
    border: 1px solid #ddd;
    padding: 6px 10px;
    text-align: left;
}

th {
    background-color: #f2f2f2;
    font-weight: bold;
}

tr:nth-child(even) {
    background-color: #fafafa;
}

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 14pt auto 6pt auto;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
}

code {
    font-family: 'Consolas', 'Courier New', monospace;
    background-color: #f4f4f4;
    padding: 2px 5px;
    font-size: 9.5pt;
    border-radius: 3px;
}

pre {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    padding: 10px;
    overflow-x: auto;
    font-size: 9pt;
    border-radius: 4px;
}

pre code {
    background-color: transparent;
    padding: 0;
}

hr {
    border: none;
    border-top: 1px solid #e0e0e0;
    margin: 20pt 0;
}

ul, ol {
    margin-bottom: 10pt;
    padding-left: 24px;
}

li {
    margin-bottom: 4px;
}
"""

full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Does Proxy-Based Data Selection Survive Contact With Reality?</title>
<style>
{css}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

html_out_path.write_text(full_html, encoding="utf-8")
print(f"Wrote HTML to {html_out_path}")

# Render to PDF using Edge Headless
edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
cmd = [
    edge_path,
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_out_path}",
    html_out_path.as_uri()
]

print("Rendering PDF with Edge...")
subprocess.run(cmd, check=True)
print(f"SUCCESS: Generated PDF at {pdf_out_path}")
