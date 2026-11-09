from docx import Document
from pathlib import Path
base=Path(r"C:\czh\codex_project\docs\overleaf\专利相关材料")
out=[]
for p in sorted(base.glob("*.docx")):
    out.append(f"\n==== {p.name} ====")
    d=Document(p)
    out.append(f"sections {len(d.sections)} paras {len(d.paragraphs)} tables {len(d.tables)} inline_shapes {len(d.inline_shapes)}")
    for i,para in enumerate(d.paragraphs):
        t=para.text.replace("\n"," ")
        if t.strip():
            out.append(f"P{i} [{para.style.name}] {t[:1000]}")
    for ti,tbl in enumerate(d.tables):
        out.append(f"TABLE {ti} rows={len(tbl.rows)} cols={len(tbl.columns)}")
        for row in tbl.rows[:5]:
            out.append(" | ".join(c.text.replace("\n"," / ")[:300] for c in row.cells))
Path(r"C:\czh\codex_project\tmp\patent_text.txt").write_text("\n".join(out),encoding="utf-8")
