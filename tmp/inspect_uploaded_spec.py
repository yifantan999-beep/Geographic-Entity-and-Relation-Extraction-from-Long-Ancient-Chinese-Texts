from docx import Document
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
src=Path(r"C:\Users\wuwei\Downloads\长文本古籍地理抽取_专利申请文件全套\3_说明书.docx")
d=Document(src)
print("paras",len(d.paragraphs),"tables",len(d.tables),"shapes",len(d.inline_shapes),"sections",len(d.sections))
ns={"m":"http://schemas.openxmlformats.org/officeDocument/2006/math","w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
with ZipFile(src) as z:
    root=etree.fromstring(z.read("word/document.xml"))
    print("OMath",len(root.xpath(".//m:oMath",namespaces=ns)),"OMathPara",len(root.xpath(".//m:oMathPara",namespaces=ns)))
for i,p in enumerate(d.paragraphs):
    if p.text.strip():
        print(f"P{i} style={p.style.name!r} text={p.text!r}")
for ti,t in enumerate(d.tables):
    print("TABLE",ti,len(t.rows),len(t.columns))
    for ri,row in enumerate(t.rows):
        for ci,cell in enumerate(row.cells):
            if cell.text.strip(): print(f"T{ti}R{ri}C{ci}={cell.text!r}")
