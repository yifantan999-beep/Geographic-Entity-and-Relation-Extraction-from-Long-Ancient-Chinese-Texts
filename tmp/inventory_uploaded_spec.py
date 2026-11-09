from docx import Document
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
src=Path(r"C:\Users\wuwei\Downloads\长文本古籍地理抽取_专利申请文件全套\3_说明书.docx")
out=Path(r"C:\czh\codex_project\tmp\uploaded_spec_inventory.txt")
d=Document(src)
lines=[]
lines.append(f"paras={len(d.paragraphs)} tables={len(d.tables)} shapes={len(d.inline_shapes)} sections={len(d.sections)}")
for i,p in enumerate(d.paragraphs):
    if p.text.strip(): lines.append(f"P{i} [{p.style.name}] {p.text}")
for ti,t in enumerate(d.tables):
    lines.append(f"\nTABLE {ti}: {len(t.rows)}x{len(t.columns)}")
    for ri,row in enumerate(t.rows):
        for ci,cell in enumerate(row.cells):
            if cell.text.strip(): lines.append(f"T{ti}R{ri}C{ci} {cell.text}")
with ZipFile(src) as z:
    xml=z.read("word/document.xml")
    root=etree.fromstring(xml)
    for name,expr in [
        ("omath",".//m:oMath"),("omathpara",".//m:oMathPara"),
        ("drawing",".//w:drawing"),("pict",".//w:pict"),
        ("object",".//w:object"),("textbox",".//w:txbxContent"),
        ("altChunk",".//w:altChunk")]:
        ns={"m":"http://schemas.openxmlformats.org/officeDocument/2006/math","w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        lines.append(f"XML {name} {len(root.xpath(expr,namespaces=ns))}")
    lines.append("zipmedia="+str([x for x in z.namelist() if x.startswith("word/media/")]))
out.write_text("\n".join(lines),encoding="utf-8")
print(out)
