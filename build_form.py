from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.style import WD_STYLE_TYPE
from pathlib import Path


OUT = Path(r"C:\czh\codex_project\工程实践总结与考核表_已填写.docx")


def set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn('w:shd'))
    if shd is None:
        shd = OxmlElement('w:shd')
        tcPr.append(shd)
    shd.set(qn('w:fill'), fill)


def set_cell_border(cell, **kwargs):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = tcPr.first_child_found_in('w:tcBorders')
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        if edge in kwargs:
            tag = 'w:{}'.format(edge)
            element = tcBorders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                tcBorders.append(element)
            for key in ['val', 'sz', 'space', 'color']:
                if key in kwargs[edge]:
                    element.set(qn('w:{}'.format(key)), str(kwargs[edge][key]))


def set_cell_margins(cell, top=100, start=120, bottom=100, end=120):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = tcPr.first_child_found_in('w:tcMar')
    if tcMar is None:
        tcMar = OxmlElement('w:tcMar')
        tcPr.append(tcMar)
    for m, v in [('top', top), ('start', start), ('bottom', bottom), ('end', end)]:
        node = tcMar.find(qn('w:' + m))
        if node is None:
            node = OxmlElement('w:' + m)
            tcMar.append(node)
        node.set(qn('w:w'), str(v))
        node.set(qn('w:type'), 'dxa')


def set_cell_width(cell, width_cm):
    tcPr = cell._tc.get_or_add_tcPr()
    tcW = tcPr.first_child_found_in('w:tcW')
    if tcW is None:
        tcW = OxmlElement('w:tcW')
        tcPr.append(tcW)
    tcW.set(qn('w:w'), str(int(width_cm * 567)))
    tcW.set(qn('w:type'), 'dxa')


def set_run_font(run, size=10.5, bold=False, color='000000'):
    run.font.name = '宋体'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    run._element.rPr.rFonts.set(qn('w:ascii'), 'Times New Roman')
    run._element.rPr.rFonts.set(qn('w:hAnsi'), 'Times New Roman')
    run.font.size = Pt(size)
    run.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)


def format_para(p, first_line=True, space_after=2, line=1.15):
    pf = p.paragraph_format
    pf.line_spacing = line
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(0)
    if first_line:
        pf.first_line_indent = Cm(0.74)
    for r in p.runs:
        set_run_font(r)


def add_cell_text(cell, text, size=10.5, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT, first_line=True):
    cell.text = ''
    parts = text.split('\n')
    for idx, part in enumerate(parts):
        p = cell.paragraphs[0] if idx == 0 else cell.add_paragraph()
        p.alignment = align
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.first_line_indent = Cm(0.74) if first_line and len(part) > 30 else Pt(0)
        run = p.add_run(part)
        set_run_font(run, size=size, bold=bold)


def style_table(table, label_col=3.1):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            set_cell_border(cell, top={'val':'single','sz':'6','color':'B7B7B7'},
                            bottom={'val':'single','sz':'6','color':'B7B7B7'},
                            left={'val':'single','sz':'6','color':'B7B7B7'},
                            right={'val':'single','sz':'6','color':'B7B7B7'})
            set_cell_width(cell, label_col if idx == 0 else 13.4)


def add_section_row(table, label, body, size=10.5):
    cells = table.add_row().cells
    cells[0].merge(cells[1])
    add_cell_text(cells[0], label, size=11, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, first_line=False)
    set_cell_shading(cells[0], 'EDEDED')
    add_cell_text(cells[1] if len(cells) > 1 else cells[0], body, size=size)


relation = (
    '本次工程实践与学位论文研究工作的关系十分紧密，研究主线均围绕自然语言中的空间信息和空间关系提取展开。'
    'GeoNER论文针对古代汉语长上下文中的地理实体识别与关系抽取问题，提出边界和类型辅助监督、跨度双仿射打分、'
    '重叠窗口对齐、非极大值抑制以及距离条件的局部—全局融合等方法，重点解决实体密集、关系端点跨窗口和误差传播问题。'
    '另一篇关于古代城市布局重建的论文进一步把文本抽取结果转化为空间图：通过大语言模型结构化提示抽取实体与方向关系，'
    '以稳定地标作为空间锚点，利用关系感知消息传递网络推断未知节点位置，并同时输出坐标和不确定性。'
    '我的工程实践将两项工作的思想串联起来，先建设面向地理文本的实体与关系抽取服务，再将抽取结果规范化为可计算的图结构，'
    '与PostgreSQL/PostGIS中的实体、坐标和空间关系关联，形成从文本、知识图谱到空间数据库的闭环。实践中的数据清洗、标注、'
    '模型微调、接口封装、空间查询和结果校验，既是论文方法的工程验证，也是学位论文研究成果面向实际应用的延伸。'
)

practice = (
    '本次工程实践以“面向历史地理文本的空间信息抽取与城市布局重建”为主线，目标是把研究生阶段学习的数据库、自然语言处理、'
    '大语言模型和地理信息系统知识，落实到一个可以运行、测试和迭代的地理信息智能处理流程中。实践首先从项目环境和数据规范入手，'
    '熟悉Linux/Python开发环境、深度学习训练流程、PostgreSQL及PostGIS空间数据库的基本操作，梳理文本、实体、关系、坐标、来源和置信度等核心对象，'
    '设计统一的数据字典、表结构和接口字段。针对古代汉语、地方志、城志及历史建筑描述中存在的异体字、断句不稳定、地名多义、实体边界模糊等问题，'
    '开展原始语料清洗、去重、分句、规范化和样本划分，并按照地名、行政区、河流、山脉、道路、建筑、公共设施等类别补充实体标注，'
    '同时标注东、西、南、北、东北等方向关系以及前后、左右、相邻等相对关系。对于长文本，按照分词结果建立带重叠的窗口，记录窗口的全局位置、'
    '实体可见范围和关系覆盖情况，避免仅按字符截断造成的标签泄漏。模型方面，复现并调整GeoNER中的跨度识别和关系抽取流程，比较基础序列标注、'
    '跨度模型、边界辅助监督、类型约束和局部—全局融合等配置，分析实体识别错误向关系抽取传递的路径。随后参考古代城市布局重建论文，'
    '设计结构化提示模板，使大语言模型只输出规定格式的实体和关系JSON，并编写解析、校验、重试和去重程序，将结果转换为异构有向空间图。'
    '在图推理环节，选取具有稳定现实位置的城门、主要街道、河道和大型建筑作为锚点，编码节点可见状态、关系类型和方向约束，'
    '实现关系感知消息传递和坐标、不确定性联合预测的实验流程。工程化方面，将抽取、图构建、空间查询和结果导出封装为服务接口，'
    '支持按文档、实体、关系类型和空间范围检索，并通过PostGIS的几何字段、空间索引、相交查询、距离查询和方向校验验证结果。'
    '最后以典型历史城市片段进行端到端测试，检查从原文输入到结构化结果、空间图和地图图层的链路是否连贯，记录异常样本、失败原因和改进方案，'
    '形成可复用的实验配置、数据处理脚本和接口说明。'
)

learning = (
    '一、项目理解与工程任务拆解。进入实践阶段后，我首先从需求和数据流角度理解项目。自然语言中的空间信息并不是孤立的地名清单，'
    '而是由实体、实体类型、关系方向、叙述顺序、距离线索、时间背景和来源证据共同构成。若只做实体识别，后续无法判断“某桥在某仓之后”'
    '究竟对应哪两个对象；若只做关系分类，又会受到实体边界和类型错误的影响。因此我把任务拆成数据准备、地理实体识别、空间关系抽取、'
    '关系规范化、空间图构建、数据库存储、服务接口和质量评估八个环节，并为每个环节定义输入、输出、异常处理和可追溯字段。'
    '在任务拆解过程中，我特别关注研究代码与生产代码的差别：实验脚本可以接受人工检查，实际服务则必须能处理空结果、重复实体、非法JSON、'
    '坐标缺失和同名异地等情况。为此我建立了统一的中间数据格式，保留原文片段、文档编号、实体起止位置、关系方向、模型版本和置信度，'
    '使每一条空间知识都能够追溯到原始证据。\n\n'
    '二、空间数据库与数据工程学习。实践中我系统学习PostgreSQL和PostGIS的使用方法。首先根据业务对象设计文档表、文本片段表、实体表、关系表、'
    '锚点表和推理结果表，明确主键、外键、唯一约束、来源字段和模型版本字段。实体表既保存名称、类型、别名和原文位置，也保存点、线或面等几何信息；'
    '关系表保存有向起点、终点、关系类型、证据文本、距离和置信度；推理结果表区分观测坐标与模型预测坐标，并记录不确定性。其次，我练习使用GiST空间索引、'
    'ST_DWithin、ST_Distance、ST_Intersects、ST_Contains等函数完成范围检索、邻近检索和几何校验，理解空间索引与普通B树索引在查询条件上的差异。'
    '对于批量导入，我将模型输出先写入临时表，经字段类型、关系方向、实体存在性和坐标范围校验后再进入正式表，避免一条异常记录中断整个批次。'
    '我还尝试将数据库通过标准服务发布为地图图层，使空间知识不再停留在训练日志或JSON文件中，而能被查询、复核和进一步分析。这个过程使我认识到，'
    '数据库设计不是模型工作的附属环节，合理的主键、版本和证据字段会直接决定结果能否复现。\n\n'
    '三、语料处理、标注与评估。针对古代汉语文本短句密集、标点缺失、异体字较多和同一地名多种写法并存的特点，我参与制定了清洗规范。'
    '清洗时保留原始文本副本，同时建立规范化文本，记录字符映射，避免直接替换导致实体位置丢失。标注时优先确定实体边界，再确定实体类型，'
    '对“某门”“某坊”“某桥”等具有类别提示但缺少完整专名的表达，根据上下文和标注指南决定是否作为实体；对嵌套或相邻实体，记录层级和冲突原因。'
    '关系标注不仅关注方向词，还关注主语省略、倒装、分句连接和“东曰”“后曰”等历史文本常见表达。为了评估模型，我分别统计实体边界F1、类型F1、'
    '严格关系F1和按距离分层的关系F1，并区分同窗口关系、跨窗口关系、长实体关系和低频关系。这样既能观察整体指标，也能定位模型在真实使用中的薄弱环节。'
    '我学习了混淆矩阵、错误样本抽样、阈值调节和消融实验的基本方法，避免只看一个综合分数。对于预测错误，我把原因归为边界错误、类型混淆、实体漏检、'
    '关系方向颠倒、无关系误报和跨窗口证据不足，并将高频错误回流到标注规范和提示模板中。\n\n'
    '四、GeoNER方法的复现与工程改造。GeoNER论文给我的直接启发是，长上下文中的实体识别和关系抽取必须同时考虑局部证据、文档级证据以及候选空间控制。'
    '我首先复现基于重叠窗口的文本切分流程，确保窗口边界只由分词长度决定，不使用金标准实体和关系信息。每个窗口记录全局词位置和可见实体，'
    '再将窗口内预测的跨度映射回文档坐标。针对重复预测，我比较了简单去重、按得分保留和非极大值抑制等策略，观察不同策略对召回率和重复率的影响。'
    '在实体阶段，我重点理解跨度双仿射打分、边界辅助监督和类型辅助监督的作用：边界信号帮助模型区分完整地名和包含地名的更长短语，类型信号减少道路、'
    '水系、行政区等相近类别的混淆。关系阶段，我按照实体类型和方向构造候选对，利用窗口内局部表示和文档级表示进行融合；对于相距较远或没有共同窗口的实体对，'
    '提高全局表示的权重，并利用长距离专家和类型兼容性约束抑制无效组合。论文报告在长上下文数据上NER-F1提升4.94个百分点、严格关系F1提升13.94个百分点，'
    '我将这些结果作为复现检查的参考，同时更关注不同距离区间和误差传播分析，而不是简单追求一个数字。工程实现中，我把推理模块改造成可配置组件，'
    '允许切换窗口长度、重叠大小、候选上限、关系阈值和类型约束，并为每次运行保存配置和模型版本。\n\n'
    '五、结构化提示与空间图构建。古代城市布局论文展示了从文本到地图的另一条路径，即先用结构化提示抽取实体和方向关系，再将实体作为节点、关系作为有向边，'
    '通过稳定地标约束未知位置。学习这一方法后，我设计了包含任务定义、输出格式、实体关系定义和示例解释四部分的提示模板，明确要求模型只返回JSON，'
    '并为东、西、南、北、东北、东南、西北、西南、前、后、左、右等表达建立统一语义。解析器对字段缺失、额外文本、重复关系和未知关系类型进行校验，'
    '必要时执行有限次数的重试，不把不合规文本直接写入数据库。图构建时，我区分已知坐标锚点和待推断节点，对相对的前、后、左、右关系按相邻约束保存，'
    '避免把相对方向误当作绝对坐标。参考SpatialRelationMPNN的思路，我理解了关系嵌入、边级消息传递、度归一化聚合、残差更新、坐标回归和不确定性预测的作用，'
    '并实现了小规模实验。损失函数同时考虑坐标误差、方向符号、角度偏差、距离先验和锚点稳定性，这让我认识到空间推理不能只优化点位误差，'
    '还要检查预测结果是否违反文本中明确的方向关系。\n\n'
    '六、接口开发、系统联调与质量控制。我围绕“原文输入—抽取服务—空间图—数据库—查询展示”的主链路编写接口和测试用例。接口层提供文本抽取、'
    '批量导入、实体检索、关系检索、空间范围查询和结果导出等功能，统一返回任务编号、状态、结果和错误信息。联调时重点测试空文本、超长文本、重复提交、'
    '模型超时、JSON解析失败、数据库连接异常和部分成功等场景。为了保证可追溯性，我在日志中记录请求时间、文本哈希、模型版本、提示模板版本、窗口配置和数据库事务状态，'
    '但不在普通日志中重复保存整篇原文。测试既包括单元测试，也包括固定样本的回归测试和端到端测试。每次模型或规则变更后，自动比较实体数、关系数、严格匹配率、'
    '无效关系比例和空间约束违例数，防止局部改进造成整体退化。通过联调我发现，研究阶段常被忽略的字段命名、编码、空值和事务提交问题，会成为服务上线后的主要故障来源。'
    '因此我逐步补充数据校验、异常回滚和结果版本化机制。\n\n'
    '七、个人学习方式与阶段性成果。除了完成代码和实验，我坚持阅读空间信息抽取、文档级关系抽取、地理信息检索、图神经网络和历史GIS方面的论文，'
    '用问题清单记录每篇论文的任务定义、数据假设、模型结构、评价指标和可迁移部分。对于不熟悉的数学表达，我先结合代码追踪张量形状，再用小数据验证公式中的每个项；'
    '对于不熟悉的地理概念，我通过数据库查询和地图图层观察实体关系的实际含义。实践阶段形成了数据字典、标注规范、窗口切分脚本、抽取接口、图构建模块、'
    '空间查询示例和错误分析记录。更重要的是，我逐渐建立了“先定义证据和约束，再选择模型；先设计可验证的中间结果，再追求端到端效果”的工作习惯。'
    '这些工作使我的学位论文研究具备了更清晰的工程落点，也为后续扩大语料、完善空间推理和开展用户试用准备了基础。'
)

gains = (
    '通过本次工程实践，我对自然语言空间信息处理有了从算法到系统的完整认识。过去我更关注模型结构和指标，实践后更加重视数据规范、证据保留、'
    '空间约束、接口稳定性和结果可追溯性。GeoNER工作使我理解了长文本实体识别、跨窗口关系抽取和误差传播之间的联系；古代城市布局重建工作则让我认识到，'
    '抽取结果只有进入空间图和数据库，才能真正支持历史地理分析。实践也提高了我使用PostgreSQL/PostGIS、组织实验和排查系统问题的能力。'
    '目前仍存在一些问题：古代文本的标注规模有限，低频关系和隐含关系的泛化能力不足；同名地名、相对方向和时间变化会给空间对齐带来歧义；'
    '大语言模型输出的稳定性、推理成本和事实可靠性还需要进一步控制；图推理结果的评价也不能只依赖坐标误差，还应结合历史证据和专家审核。'
    '后续我将扩大多来源语料，完善实体链接和证据分级机制，继续优化长距离关系建模与不确定性校准，并通过更多真实案例检验系统的可用性。'
)


def build():
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Cm(1.45)
    sec.bottom_margin = Cm(1.45)
    sec.left_margin = Cm(1.55)
    sec.right_margin = Cm(1.55)
    sec.header_distance = Cm(0.7)
    sec.footer_distance = Cm(0.7)

    normal = doc.styles['Normal']
    normal.font.name = '宋体'
    normal._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    normal.font.size = Pt(10.5)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(7)
    r = title.add_run('全日制专业学位研究生工程实践总结与考核表')
    set_run_font(r, size=16, bold=True)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rr = subtitle.add_run('（自然语言空间信息与关系提取方向）')
    set_run_font(rr, size=10.5)
    subtitle.paragraph_format.space_after = Pt(8)

    info = doc.add_table(rows=4, cols=4)
    info.alignment = WD_TABLE_ALIGNMENT.CENTER
    info.autofit = False
    widths = [2.5, 5.7, 2.5, 5.8]
    values = [
        ('研究生姓名、电话', '________________', '学号', '________________'),
        ('类型', '□ 普通  □ 定向（实践单位名称：________________）  □ 非全', '工程领域', '软件工程'),
        ('实践单位名称', '________________', '实践单位性质', '________________'),
        ('实践单位联系人、电话', '________________', '实践指导教师姓名、电话', '________________'),
    ]
    for i, row in enumerate(info.rows):
        for j, cell in enumerate(row.cells):
            set_cell_width(cell, widths[j])
            set_cell_margins(cell, top=85, bottom=85, start=100, end=100)
            set_cell_border(cell, top={'val':'single','sz':'6','color':'B7B7B7'}, bottom={'val':'single','sz':'6','color':'B7B7B7'}, left={'val':'single','sz':'6','color':'B7B7B7'}, right={'val':'single','sz':'6','color':'B7B7B7'})
            if j in (0,2):
                set_cell_shading(cell, 'EDEDED')
                add_cell_text(cell, values[i][j], size=9.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, first_line=False)
            else:
                add_cell_text(cell, values[i][j], size=9.5, first_line=False)

    promise = doc.add_table(rows=1, cols=1)
    promise.alignment = WD_TABLE_ALIGNMENT.CENTER
    promise.autofit = False
    c = promise.cell(0,0)
    set_cell_width(c, 16.5)
    set_cell_margins(c, top=110, bottom=110)
    set_cell_border(c, top={'val':'single','sz':'6','color':'B7B7B7'}, bottom={'val':'single','sz':'6','color':'B7B7B7'}, left={'val':'single','sz':'6','color':'B7B7B7'}, right={'val':'single','sz':'6','color':'B7B7B7'})
    add_cell_text(c, '研究生承诺：本人严格遵守学校及校外实践单位有关规定，注意人身和财产安全，并做到在实践期间经常与学校保持联系。\n\n本人签字：________________    日期：________年____月____日', size=10, first_line=False)

    main = doc.add_table(rows=0, cols=2)
    main.alignment = WD_TABLE_ALIGNMENT.CENTER
    main.autofit = False
    for lab, body in [
        ('与学位论文研究工作的关系', relation),
        ('实践内容（不少于500字）', practice),
        ('本人学习与工作的内容（不少于3000字）', learning),
        ('收获、体会与存在的问题（不少于200字）', gains),
    ]:
        row = main.add_row()
        row.cells[0].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        row.cells[1].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
        set_cell_width(row.cells[0], 3.1)
        set_cell_width(row.cells[1], 13.4)
        for cell in row.cells:
            set_cell_margins(cell, top=120, bottom=120, start=120, end=120)
            set_cell_border(cell, top={'val':'single','sz':'6','color':'B7B7B7'}, bottom={'val':'single','sz':'6','color':'B7B7B7'}, left={'val':'single','sz':'6','color':'B7B7B7'}, right={'val':'single','sz':'6','color':'B7B7B7'})
        set_cell_shading(row.cells[0], 'EDEDED')
        add_cell_text(row.cells[0], lab, size=10.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, first_line=False)
        add_cell_text(row.cells[1], body, size=10.2, first_line=True)

    sign = doc.add_table(rows=2, cols=4)
    sign.alignment = WD_TABLE_ALIGNMENT.CENTER
    sign.autofit = False
    sign_values = [
        ('实践单位意见', '单位负责人意见：\n\n单位负责人签字：____________\n（单位盖章）\n日期：________年____月____日', '校内导师意见', '导师意见：\n\n导师签字：____________\n日期：________年____月____日'),
        ('学院考核意见', '考核结论：□ 优秀  □ 良好  □ 合格  □ 不合格\n学院负责人签字：____________\n（学院盖章）\n日期：________年____月____日', '备注', '________________'),
    ]
    for i,row in enumerate(sign.rows):
        for j,cell in enumerate(row.cells):
            set_cell_width(cell, [2.5,5.7,2.5,5.8][j])
            set_cell_margins(cell, top=100, bottom=100, start=100, end=100)
            set_cell_border(cell, top={'val':'single','sz':'6','color':'B7B7B7'}, bottom={'val':'single','sz':'6','color':'B7B7B7'}, left={'val':'single','sz':'6','color':'B7B7B7'}, right={'val':'single','sz':'6','color':'B7B7B7'})
            if j in (0,2):
                set_cell_shading(cell, 'EDEDED')
                add_cell_text(cell, sign_values[i][j], size=10, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, first_line=False)
            else:
                add_cell_text(cell, sign_values[i][j], size=9.8, first_line=False)

    # Allow the narrative rows to flow across pages; this is essential for the
    # long learning-and-work section and prevents a tall cell from overflowing.

    doc.core_properties.title = '全日制专业学位研究生工程实践总结与考核表'
    doc.core_properties.subject = '自然语言空间信息与关系提取方向工程实践总结'
    doc.core_properties.author = ''
    doc.save(OUT)


if __name__ == '__main__':
    build()
