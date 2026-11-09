from docx import Document
from docx.shared import Mm, Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pathlib import Path
import shutil, os

OUT = Path(r"C:\czh\codex_project\docs\overleaf\专利相关材料")
FIG = Path(r"C:\czh\codex_project\tmp\paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

TITLE = "面向长上下文古汉语地理信息抽取的方法、系统、设备及存储介质"

def set_run_font(run, name="宋体", size=12, bold=False, italic=False):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Times New Roman")
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic

def configure_doc(doc):
    sec = doc.sections[0]
    sec.page_width = Mm(210)
    sec.page_height = Mm(297)
    sec.left_margin = Mm(25)
    sec.right_margin = Mm(15)
    sec.top_margin = Mm(25)
    sec.bottom_margin = Mm(15)
    sec.header_distance = Mm(15)
    sec.footer_distance = Mm(2)
    normal = doc.styles["Normal"]
    normal.font.name = "宋体"
    normal._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
    normal._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.first_line_indent = Mm(8)
    for name in ["Title","Heading 1","Heading 2","Heading 3","Caption"]:
        try:
            st = doc.styles[name]
        except KeyError:
            st = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        st.font.name = "宋体"
        st._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
        st._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Times New Roman")
        st._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Times New Roman")
        st.font.size = Pt(12)
        st.paragraph_format.line_spacing = 1.5
    doc.styles["Title"].font.size = Pt(12)
    doc.styles["Title"].font.bold = True
    doc.styles["Title"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.styles["Title"].paragraph_format.first_line_indent = Mm(0)
    doc.styles["Heading 1"].font.bold = True
    doc.styles["Heading 1"].paragraph_format.first_line_indent = Mm(0)
    doc.styles["Heading 1"].paragraph_format.space_before = Pt(6)
    doc.styles["Heading 1"].paragraph_format.space_after = Pt(0)
    doc.styles["Heading 2"].font.bold = True
    doc.styles["Heading 2"].paragraph_format.first_line_indent = Mm(0)
    doc.styles["Heading 2"].paragraph_format.space_before = Pt(4)
    doc.styles["Heading 2"].paragraph_format.space_after = Pt(0)
    doc.styles["Caption"].font.size = Pt(11)
    doc.styles["Caption"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.styles["Caption"].paragraph_format.first_line_indent = Mm(0)
    # Footer page number
    footer = sec.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fldChar1 = OxmlElement("w:fldChar"); fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText"); instrText.set(qn("xml:space"), "preserve"); instrText.text = " PAGE "
    fldChar2 = OxmlElement("w:fldChar"); fldChar2.set(qn("w:fldCharType"), "end")
    run._r.append(fldChar1); run._r.append(instrText); run._r.append(fldChar2)
    set_run_font(run, "宋体", 9)

def add_p(doc, text="", indent=True, align=None, bold=False, size=12, style=None):
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    if indent and style not in ["Title","Heading 1","Heading 2","Heading 3","Caption"]:
        p.paragraph_format.first_line_indent = Mm(8)
    else:
        p.paragraph_format.first_line_indent = Mm(0)
    p.paragraph_format.line_spacing = 1.5
    r = p.add_run(text)
    set_run_font(r, "宋体", size, bold)
    return p

def add_heading(doc, text, level=1):
    return add_p(doc, text, indent=False, style=f"Heading {level}", bold=True)

def add_eq(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Mm(0)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(text)
    set_run_font(r, "Cambria Math", 11)
    return p

def add_title(doc, text=TITLE):
    add_p(doc, text, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=12, style="Title")

def add_claim(doc, n, text):
    p=doc.add_paragraph()
    p.paragraph_format.first_line_indent = Mm(8)
    p.paragraph_format.line_spacing = 1.5
    r=p.add_run(f"{n}．{text}")
    set_run_font(r,"宋体",12)
    return p

def save(doc, path):
    doc.core_properties.title = TITLE
    doc.core_properties.subject = "专利申请文件"
    doc.core_properties.author = ""
    doc.save(path)

# 2 Claims
doc=Document(); configure_doc(doc)
claims = [
"一种面向长上下文古汉语地理信息抽取的方法，其特征在于，包括以下步骤：",
"步骤1：获取包含古汉语历史文本、地理实体标注和空间关系标注的训练数据，并按照分词结果将每一篇长文本划分为具有重叠区域的多个编码窗口，保存每个窗口的全局词位置范围以及所属源文本标识；",
"步骤2：将所述编码窗口输入共享的上下文编码器，在第一阶段通过跨度识别解码器生成实体候选，所述跨度识别解码器根据实体起始边界、结束边界和实体类型对候选跨度进行评分，并利用跨度级监督、实体类型辅助监督和边界辅助监督训练；",
"步骤3：将各窗口内得到的实体候选映射到原始长文本的全局坐标，对具有相同边界的候选进行合并，并按照跨度置信度、类型置信度、窗口支持度和类型间隔进行排序，通过非极大值抑制保留去重后的扁平化实体集合；",
"步骤4：基于所述去重后的实体集合构造同一源文本内的有向实体对，在第二阶段冻结或固定第一阶段的局部编码器，对各编码窗口重新编码，分别形成实体表示和窗口表示，并将实体类型、实体方向和实体间距离信息加入实体对表示；",
"步骤5：针对每一个实体对计算窗口条件证据注意力，将包含实体端点的窗口和位于两个实体端点之间的窗口作为软证据目标的高权重窗口，聚合得到实体对的文档级全局关系表示；",
"步骤6：分别计算所述实体对的文档级全局关系得分和在至少一个窗口内共同出现时的局部关系得分，当局部关系得分存在时，通过输入实体间距离、端点类型置信度、证据注意力集中度、局部得分可用性以及局部和全局关系得分的门控函数进行局部与全局融合；当实体对未在同一窗口内共同出现时，使用文档级全局关系得分作为回退得分；",
"步骤7：对融合后的多标签关系概率进行阈值解码，输出包含实体边界、实体类型、关系方向和空间关系类型的结构化古汉语地理知识。",
"根据权利要求1所述的方法，其特征在于，步骤1中编码窗口采用半开区间 W_k=[a_k,b_k)，窗口经分词和特殊标记后的长度不超过预设编码长度，下一窗口的起始位置按照 a_(k+1)=max(a_k+1,b_k-o) 计算，其中 o 为窗口重叠长度；每个窗口同时记录完整可见的实体和关系标识，以避免跨源文本构造虚假关系。",
"根据权利要求1所述的方法，其特征在于，步骤2中的上下文编码器为中文预训练语言模型，并可包括层级块边界上下文细化模块；所述细化模块通过局部卷积路径、块级均值表示和逐词门控，将长文本的块级上下文广播到词位置，同时保留相邻实体边界的局部特征。",
"根据权利要求1所述的方法，其特征在于，步骤2中的跨度候选(s,e,t)的评分为：q(s,e,t)=q_base(s,e,t)+γ(s,e,t)[p_s^T U_t p_e+r_t(e-s)]，其中 q_base 为基础跨度得分，p_s 和 p_e 为起止位置的低秩投影，U_t 为实体类型对应的双线性矩阵，r_t(e-s) 为跨度宽度嵌入，γ 为可学习门控系数。",
"根据权利要求1所述的方法，其特征在于，所述实体类型辅助监督对每一个真实实体跨度计算带类别权重的交叉熵或焦点损失；所述边界辅助监督分别预测每个词位置是否为实体起点和实体终点，并对真实边界、边界邻近位置以及真实实体内部的非边界位置设置不同的加权二元交叉熵。",
"根据权利要求1所述的方法，其特征在于，步骤3中的非极大值抑制采用任意重叠抑制规则：候选实体按照聚合选择分数由高到低排序，当候选实体与已保留实体发生边界重叠时将其删除，从而得到无重复且不重叠的全局实体集合；所述聚合选择分数至少包含窗口支持度、跨度置信度和实体类型置信度。",
"根据权利要求1所述的方法，其特征在于，步骤5中，对实体对(i,j)和窗口W_k的注意力分数为 Q(u_ij)^T K(w_k)/sqrt(d_h) 加上窗口是否包含实体i和实体j的覆盖指示项，并通过对全部窗口执行 softmax 得到注意力 a_ijk；所述全局表示为 v_ij=sum_k a_ijk V(w_k)。",
"根据权利要求1所述的方法，其特征在于，步骤5中的软证据目标由端点覆盖项和端点中心之间的窗口中心项组成：g_ijk=I[W_k包含i]+I[W_k包含j]+0.25I[c_k位于两个实体中心之间]，并将 g_ijk 归一化为目标分布；对具有至少一个标注关系的实体对计算该目标分布与注意力分布之间的 KL 散度。",
"根据权利要求1所述的方法，其特征在于，步骤6中的距离条件模块将实体中心距离划分为[0,32)、[32,64)、[64,128)和[128,+∞)四个距离区间，为每个区间设置一个专家分类器，并在距离大于或等于128个词位置时叠加长距离关系头的残差得分；各专家得分通过距离门控权重加权求和。",
"根据权利要求1所述的方法，其特征在于，训练阶段的关系目标函数包括非负正例-未标注风险、证据注意力损失、距离加权正例损失、长距离排序损失、实体类型兼容性损失、正例焦点损失和距离专家损失的加权和；其中未标注实体对在正例-未标注风险中作为未标注样本处理。",
"根据权利要求1所述的方法，其特征在于，所述正例焦点损失仅对标注为正的实体对-关系三元组计算，并按照 (1-p_ijr)^γ softplus(-z_ijr) 进行调制；当实体中心距离大于或等于128个词位置时，对该正例乘以预设长距离权重，未标注实体对不在该辅助损失中作为硬负例使用。",
"根据权利要求1所述的方法，其特征在于，还包括可选的实体对存在性分支，所述分支输出实体对是否存在任一关系的二元概率，并利用二元交叉熵与主关系目标共同训练。",
"一种用于实现权利要求1至10任一项所述方法的面向长上下文古汉语地理信息抽取系统，其特征在于，包括：长文本窗口构建模块，用于构造重叠编码窗口并维护全局坐标；边界和类型感知实体识别模块，用于识别、分类和合并地理实体；实体对构建模块，用于在全局实体集合上构造有向实体对；文档级证据聚合模块，用于计算窗口条件证据注意力；距离条件关系推理模块，用于计算类型兼容性、距离专家和长距离关系得分；局部全局融合模块，用于在局部关系得分可用时进行融合并在跨窗口场景下执行全局回退；以及结果输出模块，用于输出结构化地理实体和空间关系。",
"根据权利要求11所述的系统，其特征在于，还包括训练管理模块，所述训练管理模块用于按照训练目标函数进行反向传播、更新模型参数、在开发数据上选择阈值并保存训练后的实体识别模型和关系抽取模型。",
"一种电子设备，其特征在于，包括处理器、存储器以及存储在所述存储器中并可由所述处理器执行的程序，所述程序被执行时实现权利要求1至10任一项所述的方法。",
"一种计算机可读存储介质，其上存储有程序，所述程序被处理器执行时实现权利要求1至10任一项所述的方法。"
]
add_title(doc)
for i,c in enumerate(claims,1):
    add_claim(doc,i,c)
save(doc, OUT/"2_权利要求书_长上下文古汉语地理信息抽取.docx")

# 3 Specification
doc=Document(); configure_doc(doc)
add_title(doc)
add_heading(doc,"技术领域")
add_p(doc,"本发明属于自然语言处理、历史文献数字化和地理信息抽取技术领域，具体涉及一种面向长上下文古汉语历史文本的地理实体识别、空间关系抽取、模型训练与结构化地理知识生成的方法、系统、电子设备及计算机可读存储介质。")
add_heading(doc,"背景技术")
add_p(doc,"古代汉语历史文献记录了古城空间布局、行政区划、水系、道路、官署、学校、园林以及地名沿革等信息，是重建历史地理和古代城市空间的重要资料。将这类非结构化叙述转化为结构化地理知识，通常需要先识别文本中的地理实体，再确定实体之间的方位、包含、邻接、连通或其他空间关系。")
add_p(doc,"古汉语具有表达简练、意合性强、句法标记有限、成分省略普遍、同一地点存在别称和异体名称等特点。空间关系有时跨越较长文本距离，或通过上下文和常识隐含表达。上述特点使实体边界、实体类型和关系端点的确定均存在较大不确定性。")
add_p(doc,"当多个来源句子按照原始文献顺序拼接为长上下文时，地理实体密度和候选实体对数量显著增加，候选关系之间产生竞争与干扰。传统固定长度滑动窗口只能在局部范围内同时保留关系两端；当两个端点落在不同窗口时，关系证据会被割裂。若直接将未标注实体对视为负例，还会把尚未观察到的关系误判为负样本，导致长距离关系召回率下降。")
add_p(doc,"现有序列标注或局部关系抽取方法通常依赖单句或固定窗口上下文，难以处理跨窗口的实体对。长文本切分和局部编码会使同一关系的端点、别称、指代和隐含连接分散在不同窗口；实体边界和类型错误还会沿流水线传递到关系抽取阶段。仅分别报告实体识别和关系抽取指标，也难以区分上游实体错误与下游关系分类错误。")
add_heading(doc,"发明内容")
add_p(doc,"针对长上下文古汉语地理信息抽取中实体候选密集、关系端点不可共同观测、局部证据割裂以及实体识别误差向关系阶段传递的问题，本发明提出一种两阶段文档级抽取方法。第一阶段采用边界和类型感知的跨度识别完成地理实体识别，并通过窗口间坐标对齐和非极大值抑制形成全局实体集合；第二阶段在全局实体对上进行关系抽取，采用窗口条件证据聚合、类型兼容性约束、距离条件专家和局部全局门控融合，处理同窗口关系和跨窗口关系。")
add_p(doc,"本发明的技术方案如下。")
add_p(doc,"第一方面，本发明公开一种面向长上下文古汉语地理信息抽取的方法，包括以下步骤：")
for t in [
"步骤一，获取古汉语历史文本及其地理实体和空间关系标注，按照分词器将文本划分为多个具有重叠区域的半开区间窗口。每个窗口保存全局词位置、源文本标识、完整可见实体和完整可见关系标识。",
"步骤二，将各窗口输入共享的中文预训练编码器。可选地，通过层级块边界上下文细化模块生成局部卷积表示和块级上下文表示，并通过逐词门控进行融合。",
"步骤三，在第一阶段进行边界和类型感知的跨度实体识别。模型由实体起点和终点边界预测、跨度基础评分、类型特定双线性残差和跨度宽度嵌入组成，得到实体边界和细粒度实体类型的联合评分。",
"步骤四，采用实体类型辅助监督和边界辅助监督训练第一阶段模型。实体类型辅助监督对真实实体跨度计算带类型权重的分类损失；边界辅助监督分别对起点和终点计算加权二元交叉熵，并对边界邻近位置及实体内部非边界位置构造困难负例。",
"步骤五，将各窗口预测的实体候选映射到长文本全局坐标，对同边界候选进行合并并计算聚合选择分数，再采用任意重叠非极大值抑制得到去重后的全局实体集合。",
"步骤六，在第二阶段基于全局实体集合构造同一源文本内的有向实体对。重新编码窗口时固定第一阶段编码器，分别进行实体视图中心性池化和窗口均值池化，将实体类型概率、实体方向、实体中心距离和窗口位置编码加入实体对表示。",
"步骤七，针对每个实体对计算窗口条件证据注意力。注意力查询由实体对表示产生，窗口键和值由窗口表示产生；包含实体任一端点的窗口获得覆盖先验，所有窗口通过 softmax 得到注意力分布。对正例实体对构造软证据目标，使端点所在窗口和两端之间的窗口获得较高权重。",
"步骤八，利用实体对表示与证据聚合表示计算文档级全局关系得分，并利用共同包含实体对的窗口计算局部关系得分。对于局部关系得分存在的实体对，根据实体距离、端点类型置信度、证据集中度、局部得分可用性、局部得分和全局得分计算关系特定门控系数；对于跨窗口实体对，直接使用全局关系得分。",
"步骤九，在关系推理中加入类型兼容性偏置、距离条件专家和长距离残差头。距离划分为[0,32)、[32,64)、[64,128)和[128,+∞)区间；各区间专家产生残差关系得分，距离至少128个词位置时启用长距离关系头。",
"步骤十，使用正例-未标注风险、证据注意力损失、距离加权正例损失、长距离排序损失、类型兼容性损失、正例焦点损失和距离专家损失训练第二阶段模型，最后对融合后的多标签关系概率进行阈值解码，输出地理实体、实体类型、关系类型、方向和全局位置。"
]:
    add_p(doc,t)
add_heading(doc,"附加技术方案")
add_p(doc,"优选地，窗口长度由分词器确定并包括特殊标记，窗口长度不超过编码器最大输入长度；相邻窗口保持预设词级重叠，且窗口边界只由分词结果和重叠参数决定，以避免根据标签反向调整窗口。")
add_p(doc,"优选地，跨度候选(s,e,t)的评分可写为：q(s,e,t)=q_base(s,e,t)+γ(s,e,t)[p_s^T U_t p_e+r_t(e-s)]。其中，q_base为基础跨度评分，p_s和p_e为起止位置的低秩投影，U_t为类型t对应的双线性矩阵，r_t为跨度宽度嵌入，γ为控制残差幅度的可学习门控。")
add_p(doc,"优选地，实体类型损失和边界损失分别为带类型权重的正例焦点分类损失以及起点、终点加权二元交叉熵。整体实体识别目标由原始跨度目标、类型辅助目标和边界辅助目标加权组成；在需要进一步区分邻近边界或相近类型时，可增加跨度间隔损失。")
add_p(doc,"优选地，实体对(i,j)的基础表示由两个实体表示、表示差的绝对值、逐元素乘积、类型投影、距离嵌入和方向嵌入拼接后经过前馈网络和层归一化得到。实体在多个窗口中出现时，按照实体边界到窗口边界的最小距离计算中心性权重，使位于窗口中部的实体视图获得更大权重。")
add_p(doc,"优选地，窗口证据注意力可写为：a_ijk=softmax_k(Q(u_ij)ᵀK(w_k)/√d_h+I[W_k包含i]+I[W_k包含j])，全局证据表示为 v_ij=Σ_k a_ijk V(w_k)。软证据目标的未归一化权重为 g_ijk=I[W_k包含i]+I[W_k包含j]+0.25I[c_k位于两个实体中心之间]，归一化后得到目标分布；当没有窗口获得目标权重时，将离两实体中心中点最近的窗口作为目标窗口。")
add_p(doc,"优选地，关系训练采用非负正例-未标注风险。对于关系类型r及其先验π_r，正例部分使用 softplus(-z_r)，未标注部分使用 softplus(z_r)并减去正例校正项，校正后的风险取非负值。该设计避免将未标注实体对直接作为确定负例。")
add_p(doc,"优选地，正例焦点损失仅在标注正例上计算，其形式为 L_focal=平均[α_r η_ij (1-p_ijr)^γ softplus(-z_ijr)]。当实体间距离不小于128个词位置时，η_ij取预设的长距离权重；否则取1。该辅助项与正例-未标注风险共同使用，未标注实体对不在该辅助项中作为硬负例。")
add_p(doc,"优选地，局部全局融合在局部关系得分不存在时使用全局分支回退，从而保留端点不在同一窗口的实体对；在局部关系得分存在时，门控系数越偏向全局分支，表示模型越依赖跨窗口证据。")
add_heading(doc,"有益效果")
for t in [
"本发明通过窗口间全局坐标对齐、实体去重和非极大值抑制控制高密度文本中的候选实体数量，降低了实体对数量增长造成的关系竞争。",
"本发明通过实体类型和边界辅助监督增强跨度表示，使实体边界、细粒度类型和后续实体对构造之间形成可传递的约束。",
"本发明通过窗口条件证据注意力和局部全局融合保留跨窗口关系端点，在端点无法共同出现在一个局部窗口时仍可利用文档级表示完成关系判断。",
"本发明通过距离条件专家、长距离关系头、距离排序和正例焦点损失增强长距离稀疏关系的识别能力，同时避免将未标注关系作为硬负例。",
"本发明输出实体边界、实体类型、关系方向和关系类型，可用于古城空间布局重建、历史地名演变分析和历史地理知识图谱构建。"
]:
    add_p(doc,t)
add_heading(doc,"附图说明")
for t in [
"图1为本发明面向长上下文古汉语地理信息抽取的总体流程示意图。",
"图2为本发明边界和类型感知跨度实体识别模块示意图。",
"图3为本发明文档级关系抽取、证据注意力和局部全局融合模块示意图。",
"图4为本发明按实体间距离划分的关系抽取性能示意图。",
"图5为本发明距离条件局部全局门控行为示意图。",
"图6为本发明在不同上下文长度和端点可见性条件下的鲁棒性示意图。"
]:
    add_p(doc,t,indent=True)
add_heading(doc,"具体实施方式")
add_p(doc,"下面结合附图和实施例对本发明作进一步说明。应当说明，以下实施例用于解释本发明，而不用于限定本发明的保护范围。")
add_heading(doc,"实施例一：长上下文古汉语地理信息抽取方法", level=2)
for t in [
"本实施例使用古代城市地理文献构成的地理信息抽取语料。语料包含古汉语文本、地理实体边界、实体类型以及实体间空间关系。将原始短句按照源文献标识进行拼接，形成长上下文数据；拼接过程中保留原始实体和关系标注，不在不同源文本之间新增关系。",
"步骤一，建立窗口清单。使用中文分词器将长文本转换为词位置序列，在不超过512个子词位置的条件下确定窗口终点，下一窗口从前一窗口末端之前的重叠位置开始。窗口清单记录全局起止位置、源文本编号以及完全位于窗口内的实体和关系编号。",
"步骤二，进行第一阶段实体识别。使用中文预训练编码器获得窗口内词表示，在需要时使用层级块边界上下文细化模块增强长文本局部上下文。实体识别解码器先预测有效的实体起点，再在满足起点约束的结束位置上生成跨度候选，候选评分同时使用基础跨度评分、类型特定双线性残差和跨度宽度嵌入。",
"步骤三，进行类型和边界辅助训练。对于每个真实实体跨度，计算实体类型辅助损失；对于每个词位置分别计算实体起点和终点损失，并对真实边界附近的偏移位置和实体内部的非边界位置设置困难负例权重。通过联合优化原始跨度损失、类型辅助损失和边界辅助损失，获得边界和类型感知的实体预测模型。",
"步骤四，进行全局实体合并。把每个窗口的实体候选转换为长文本全局坐标，按实体起止位置进行分组；利用窗口支持数、跨度置信度、类型置信度和类型间隔计算聚合选择分数。候选实体按选择分数排序，对发生任意边界重叠的低分候选执行非极大值抑制，得到不重复的全局实体集合。",
"步骤五，进行第二阶段关系抽取。对全局实体集合中的同源有向实体对重新构造实体表示和窗口表示。实体在多个窗口出现时按照中心性权重池化，窗口表示采用均值池化。对每个实体对计算类型、方向和距离条件的基础表示，并通过查询与窗口键之间的相似度获得窗口注意力。",
"步骤六，进行局部全局融合。对于至少在一个窗口共同出现的实体对，保留各共同窗口的最大局部关系概率；对于未共同出现的实体对，局部关系概率标记为不可用。融合门控输入实体距离、类型置信度、证据集中度、局部和全局关系得分及其可用性。若局部得分不可用，则直接采用全局关系得分；若局部得分可用，则按门控系数融合局部和全局得分。",
"步骤七，进行距离条件推理和输出。将实体中心距离分为四个区间，分别通过距离专家产生关系残差，并在长距离区间叠加长距离关系头。对每种空间关系类型计算最终概率，按照开发集确定的关系阈值进行多标签解码，输出实体边界、实体类型、关系方向和关系类别，形成地理知识三元组或有向关系图。"
]:
    add_p(doc,t)
add_heading(doc,"实施例二：系统实施方式", level=2)
add_p(doc,"本实施例提供一种用于执行实施例一方法的系统，包括：窗口构建模块、共享编码模块、实体识别模块、实体全局合并模块、实体对构建模块、证据聚合模块、距离条件关系推理模块、局部全局融合模块、训练管理模块和结果输出模块。")
for t in [
"窗口构建模块用于读取古汉语历史文本、执行分词、生成重叠编码窗口、维护全局坐标和源文本标识；",
"共享编码模块用于输出各窗口的上下文词表示，并可执行层级块边界上下文细化；",
"实体识别模块用于完成跨度候选生成、边界预测、实体类型预测以及类型和边界辅助监督训练；",
"实体全局合并模块用于将窗口坐标映射到全局坐标、合并重复候选并执行非极大值抑制；",
"实体对构建模块用于基于全局实体集合构造同源有向实体对，并计算实体类型、方向和距离特征；",
"证据聚合模块用于计算实体对条件的窗口注意力和全局证据表示；",
"距离条件关系推理模块用于计算类型兼容性、距离专家和长距离关系头的关系得分；",
"局部全局融合模块用于融合共同窗口中的局部关系证据和文档级全局关系证据，并在跨窗口情况下执行全局回退；",
"结果输出模块用于按照关系阈值输出地理实体、空间关系及其全局位置，并可生成用于历史地理分析的结构化关系图。"
]:
    add_p(doc,t)
add_heading(doc,"实施例三：训练与评价方式", level=2)
add_p(doc,"在一个实施例中，采用古代临安地理文献构建原始数据集和长上下文派生数据集。原始数据集保留单句或单文档样本，长上下文数据集将同一源文献中的多个句子拼接为长文本。编码窗口的最大长度为512个子词位置，长上下文样本可超过1024个词位置。训练阶段仅在同一源文本内构造实体对，测试阶段分别报告严格实体识别、边界识别、严格关系抽取、匹配实体关系抽取和距离分桶关系抽取指标。")
add_p(doc,"在另一个实施例中，采用实体识别预测结果进行端到端评价，并设置实体边界类型校正和真实实体输入两类预言实验，用于区分实体识别错误与关系分类错误的影响；同时按照实体中心距离和端点是否在同一窗口共同出现进行分层误差分析。上述训练和评价设置均不改变本发明的模型流程，可用于验证跨窗口关系的可见性和长距离关系性能。")
add_p(doc,"本发明还可以通过计算机程序实现，所述程序存储在计算机可读存储介质中，由处理器执行时完成上述方法步骤。以上实施方式仅为本发明的优选实施例，凡在本发明的精神和原则内所作的等同替换、简单组合或改进，均应包含在本发明的保护范围之内。")
save(doc, OUT/"3_说明书_长上下文古汉语地理信息抽取.docx")

# 4 Drawings
doc=Document(); configure_doc(doc)
add_title(doc,"说明书附图")
figs=[
("fig1_overview.png","图1  面向长上下文古汉语地理信息抽取的总体流程示意图"),
("fig2_ner.png","图2  边界和类型感知跨度实体识别模块示意图"),
("fig3_re.png","图3  文档级关系抽取、证据注意力与局部全局融合模块示意图"),
("fig_distance_performance.png","图4  不同实体间距离区间的严格关系抽取性能示意图"),
("fig_global_gate_behavior.png","图5  距离条件局部全局门控行为示意图"),
("fig_long_context_robustness.png","图6  不同上下文长度和端点可见性条件下的鲁棒性示意图"),
]
for idx,(fn,cap) in enumerate(figs):
    if idx>0:
        doc.add_page_break()
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.first_line_indent=Mm(0)
    p.add_run().add_picture(str(FIG/fn), width=Mm(175))
    cp=doc.add_paragraph(style="Caption"); cp.alignment=WD_ALIGN_PARAGRAPH.CENTER; cp.paragraph_format.first_line_indent=Mm(0)
    r=cp.add_run(cap); set_run_font(r,"宋体",11,True)
save(doc, OUT/"4_说明书附图_长上下文古汉语地理信息抽取.docx")

# 5 Abstract
doc=Document(); configure_doc(doc)
abstract=("本发明公开一种面向长上下文古汉语地理信息抽取的方法、系统、电子设备及存储介质。该方法将古汉语历史文本划分为具有重叠区域的编码窗口，第一阶段通过边界和类型感知的跨度识别完成地理实体识别，并利用实体类型辅助监督、边界辅助监督、全局坐标合并和非极大值抑制得到去重后的实体集合；第二阶段在全局实体对上计算窗口条件证据注意力，融合实体类型、方向和距离信息，通过距离条件专家及局部全局门控处理同窗口和跨窗口空间关系，在局部证据不可用时采用文档级全局回退；训练目标结合正例未标注风险、证据注意力、距离加权、长距离排序、类型兼容性、正例焦点和距离专家损失。该方法能够降低长文本高密度实体竞争，缓解固定窗口导致的关系端点不可见和证据割裂问题，提升古城空间布局重建、历史地名演变分析和历史地理知识图谱构建的准确性。")
add_p(doc,abstract,indent=True)
save(doc, OUT/"5_说明书摘要_长上下文古汉语地理信息抽取.docx")
print("created")
