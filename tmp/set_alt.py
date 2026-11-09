from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path
from lxml import etree
p=Path(r"C:\czh\codex_project\docs\overleaf\专利相关材料\4_说明书附图_长上下文古汉语地理信息抽取.docx")
tmp=p.with_suffix(".tmp.docx")
ns={"wp":"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"}
alts=[
"图1 面向长上下文古汉语地理信息抽取的总体流程示意图",
"图2 边界和类型感知跨度实体识别模块示意图",
"图3 文档级关系抽取、证据注意力与局部全局融合模块示意图",
"图4 不同实体间距离区间的严格关系抽取性能示意图",
"图5 距离条件局部全局门控行为示意图",
"图6 不同上下文长度和端点可见性条件下的鲁棒性示意图",
]
with ZipFile(p,"r") as zin, ZipFile(tmp,"w",ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data=zin.read(item.filename)
        if item.filename=="word/document.xml":
            root=etree.fromstring(data)
            nodes=root.xpath(".//wp:docPr",namespaces=ns)
            for i,node in enumerate(nodes[:len(alts)]):
                node.set("descr",alts[i]); node.set("title",alts[i])
            data=etree.tostring(root,xml_declaration=True,encoding="UTF-8",standalone=True)
        zout.writestr(item,data)
tmp.replace(p)
print("alt text set",len(alts))
