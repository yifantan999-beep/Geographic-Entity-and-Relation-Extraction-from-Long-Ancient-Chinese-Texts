from docx import Document
from docx.oxml.ns import qn
from pathlib import Path
base=Path(r"C:\czh\codex_project\docs\overleaf\专利相关材料")
for fname in ["2_权利要求书.docx","3_说明书.docx","4_说明书附图.docx","5_说明书摘要.docx"]:
    p=base/fname; d=Document(p)
    print("\n==",fname,"==")
    for si,s in enumerate(d.sections):
        print("section",si,"page",s.page_width,s.page_height,"margins",s.left_margin,s.right_margin,s.top_margin,s.bottom_margin,"header",s.header_distance,"footer",s.footer_distance)
    for st in ["Normal","Title","Heading 1","Caption"]:
        try:
            s=d.styles[st]
            print("style",st,"font",s.font.name,s.font.size,"bold",s.font.bold,"align",s.paragraph_format.alignment,"space",s.paragraph_format.space_before,s.paragraph_format.space_after,"line",s.paragraph_format.line_spacing)
        except Exception as e: pass
    for i,para in enumerate(d.paragraphs[:10]):
        if para.text.strip() or i<4:
            f=para.runs[0].font if para.runs else None
            print("para",i,"text",para.text[:50],"align",para.alignment,"indent",para.paragraph_format.first_line_indent,para.paragraph_format.left_indent,"space",para.paragraph_format.space_before,para.paragraph_format.space_after,"line",para.paragraph_format.line_spacing,"font",f.name if f else None,f.size if f else None,"bold",f.bold if f else None)
    if d.inline_shapes:
        for sh in d.inline_shapes:
            print("shape",sh.width,sh.height)
