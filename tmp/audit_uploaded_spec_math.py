from pathlib import Path
from zipfile import ZipFile
from lxml import etree
import re
import sys
sys.path.insert(0, r"C:\czh\codex_project\tmp")
from convert_uploaded_spec_math import INLINE_PATTERN
src=Path(r"C:\Users\wuwei\Downloads\长文本古籍地理抽取_专利申请文件全套\3_说明书.docx")
out=Path(sys.argv[1]) if len(sys.argv)>1 else Path(r"C:\czh\codex_project\docs\overleaf\专利相关材料\3_说明书_公式已转为Word公式.docx")
NS={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main","m":"http://schemas.openxmlformats.org/officeDocument/2006/math"}
def parse(path):
    with ZipFile(path) as z:
        root=etree.fromstring(z.read("word/document.xml"))
    body=root.find(".//{"+NS["w"]+"}body")
    return root,body
def text_xml(node):
    return "".join(node.xpath(".//w:t/text()|.//m:t/text()",namespaces=NS))
def norm(s):
    s=re.sub(r"[_^]","",s)
    s=s.replace("{","").replace("}","").replace("'","′")
    return re.sub(r"\s+"," ",s).strip()
sroot,sbody=parse(src); oroot,obody=parse(out)
sp=sbody.xpath(".//w:p",namespaces=NS); op=obody.xpath(".//w:p",namespaces=NS)
s_tables=sbody.xpath(".//w:tbl",namespaces=NS); o_tables=obody.xpath(".//w:tbl",namespaces=NS)
s_cells=sbody.xpath(".//w:tc",namespaces=NS); o_cells=obody.xpath(".//w:tc",namespaces=NS)
print("paragraphs",len(sp),len(op),"tables",len(s_tables),len(o_tables),"cells",len(s_cells),len(o_cells))
print("equation objects",len(oroot.xpath(".//m:oMath",namespaces=NS)),"subscripts",len(oroot.xpath(".//m:sSub|.//m:sSubSup",namespaces=NS)),"superscripts",len(oroot.xpath(".//m:sSup|.//m:sSubSup",namespaces=NS)))
errs=[]
for i,(a,b) in enumerate(zip(sp,op)):
    if a.xpath("./ancestor::w:tc[1]",namespaces=NS):
        # Formula cells are checked via equation-node and numbered row counts below.
        if b.xpath(".//m:oMath",namespaces=NS): continue
        # Numbering/other table text must remain identical.
        if text_xml(a)!=text_xml(b): errs.append((i,"table text",text_xml(a),text_xml(b)))
        continue
    original="".join(a.xpath(".//w:t/text()",namespaces=NS))
    remaining="".join(b.xpath("./w:r/w:t/text()",namespaces=NS))
    # Non-math prose remains exactly as authored. Converted notation lives only in m:t.
    original_nonmath=INLINE_PATTERN.sub("",original)
    if original_nonmath != remaining:
        errs.append((i,"body mismatch",original_nonmath[:150],remaining[:150]))
print("nonmath mismatches",len(errs))
for e in errs[:10]:print("MISMATCH",e)
numbered=[]
for t in obody.xpath(".//w:tbl",namespaces=NS):
    for row in t.xpath("./w:tr",namespaces=NS):
        cells=row.xpath("./w:tc",namespaces=NS)
        if len(cells)>=2 and re.fullmatch(r"\(\d+\)",text_xml(cells[1]).strip()):
            numbered.append((text_xml(cells[1]).strip(),len(cells[0].xpath(".//m:oMath",namespaces=NS))))
print("numbered equation rows",len(numbered),"all have one math object",all(n==1 for _,n in numbered),"numbers", [n for n,_ in numbered])
# Residual notation left in non-math runs
res=[]
for i,p in enumerate(op):
    if p.xpath(".//m:oMath",namespaces=NS): continue
    txt="".join(p.xpath("./w:r/w:t/text()",namespaces=NS))
    if re.search(r"_[{A-Za-z0-9]|\^[{A-Za-z0-9]|[σγλπρφακωφθηδβμντ⊙∈∋∑√≤≥≠]",txt):
        res.append((i,txt))
print("residual math-like paragraphs",len(res))
for x in res[:12]:print("RESIDUAL",x)
