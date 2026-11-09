from docx import Document
from pathlib import Path
import re
src=Path(r"C:\Users\wuwei\Downloads\长文本古籍地理抽取_专利申请文件全套\3_说明书.docx")
d=Document(src)
pat=re.compile(r"""(?:
    [A-Za-z][A-Za-z0-9]*'(?:_\{[^}]+\}|_[A-Za-z0-9]+)? |
    [A-Za-z][A-Za-z0-9]*(?:_\{[^}]+\}|_[A-Za-z0-9]+)(?:\^\{[^}]+\}|\^[A-Za-z0-9]+)? |
    [A-Za-z][A-Za-z0-9]*\^\{[^}]+\} |
    [A-Za-z]′ |
    O\([A-Za-z]\^[0-9]+\) |
    [σγλπρφακωφθ] |
    [⊙∈∋∑√≤≥≠]
)""",re.X)
lines=[]
for i,p in enumerate(d.paragraphs):
    if p.text.strip():
        toks=[]
        for m in pat.finditer(p.text):
            tok=m.group()
            if tok not in toks:toks.append(tok)
        if toks:lines.append(f"P{i}: {' | '.join(toks)}")
for ti,t in enumerate(d.tables):
    for ri,row in enumerate(t.rows):
        text=" || ".join(c.text for c in row.cells)
        toks=[]
        for m in pat.finditer(text):
            tok=m.group()
            if tok not in toks:toks.append(tok)
        if toks:lines.append(f"T{ti}R{ri}: {' | '.join(toks)}")
Path(r"C:\czh\codex_project\tmp\math_tokens.txt").write_text("\n".join(lines),encoding="utf-8")
