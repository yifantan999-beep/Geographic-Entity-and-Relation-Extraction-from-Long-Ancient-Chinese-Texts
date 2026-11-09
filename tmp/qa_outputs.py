from docx import Document
from pathlib import Path
base=Path(r"C:\czh\codex_project\docs\overleaf\专利相关材料")
files=[
"2_权利要求书_长上下文古汉语地理信息抽取.docx",
"3_说明书_长上下文古汉语地理信息抽取.docx",
"4_说明书附图_长上下文古汉语地理信息抽取.docx",
"5_说明书摘要_长上下文古汉语地理信息抽取.docx",
]
for f in files:
    p=base/f
    d=Document(p)
    print(f, "OK", "paragraphs",len(d.paragraphs),"images",len(d.inline_shapes),"tables",len(d.tables))
    if f.startswith("2_"):
        for i,para in enumerate(d.paragraphs):
            print(i,para.text[:80])
    if f.startswith("4_"):
        print("captions", [x.text for x in d.paragraphs if x.style.name=="Caption"])
