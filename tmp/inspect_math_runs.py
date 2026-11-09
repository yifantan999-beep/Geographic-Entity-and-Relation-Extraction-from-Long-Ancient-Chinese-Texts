from zipfile import ZipFile
from lxml import etree
p=r"C:\Users\wuwei\Downloads\长文本古籍地理抽取_专利申请文件全套\3_说明书.docx"
ns={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
with ZipFile(p) as z:
 root=etree.fromstring(z.read("word/document.xml"))
 body=root.find(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}body")
 for pi,pel in enumerate(body.findall("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p")):
  txt="".join(pel.xpath(".//w:t/text()",namespaces=ns))
  if any(c in txt for c in "_^σγλπρφακωφθ⊙≥∈∋∑√") or "E'" in txt:
   runs=[]
   for rr in pel.xpath("./w:r",namespaces=ns):
    t="".join(rr.xpath(".//w:t/text()",namespaces=ns))
    if t:runs.append(t)
   print(pi,"runs",len(runs)," | ".join(runs))
