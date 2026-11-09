from docx import Document
from pathlib import Path
base=Path(r"C:\czh\codex_project\docs\overleaf\专利相关材料")
for p in sorted(base.glob("*长上下文古汉语地理信息抽取.docx")):
    d=Document(p)
    print("\n==",p.name,"==",len(d.paragraphs),len(d.tables),len(d.inline_shapes),len(d.sections))
    for i,para in enumerate(d.paragraphs[:15]):
        if para.text.strip(): print(i,para.text[:180])
