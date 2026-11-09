from __future__ import annotations

import copy
import hashlib
import re
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


SOURCE = Path(r"C:\Users\wuwei\Downloads\长文本古籍地理抽取_专利申请文件全套\3_说明书.docx")
OUTPUT = Path(r"C:\czh\codex_project\docs\overleaf\专利相关材料\3_说明书_公式已转为Word公式.docx")
DOCUMENT_XML = "word/document.xml"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS, "m": M_NS}
W = lambda name: f"{{{W_NS}}}{name}"
M = lambda name: f"{{{M_NS}}}{name}"
etree.register_namespace("w", W_NS)
etree.register_namespace("m", M_NS)

FUNCTIONS = {
    "BCEWithLogits", "Sigmoid", "softmax", "softplus", "max", "min",
    "log", "exp", "LN", "MLP", "FFN", "argmax",
}
ROOTS = {
    "L", "q", "p", "U", "r", "w", "u", "y", "G", "H", "m", "V",
    "v", "k", "e", "T", "d", "c", "a", "I", "B", "z", "E", "f",
    "Q", "K", "h", "l", "x", "b", "g", "i", "s", "t", "n", "o",
    "P", "R", "theta", "rho", "eta", "delta", "alpha", "gamma", "lambda",
    "pi", "phi", "kappa", "omega", "sigma", "tau", "beta", "mu", "nu",
}
GREEK = set("σγλπρφακωφθηδβμντ")

INLINE_PATTERN = re.compile(
    r"O\(m\^2\)"
    r"|x=\(x_1,…,x_n\)"
    r"|W_k=\[a_k,b_k\)"
    r"|a_\{k\+1\}=max\(a_k\+1,\s*b_k-o\)"
    r"|d_\{ij\}=\|c_i-c_j\|"
    r"|I_\{ij\}=\[min\(c_i,c_j\),\s*max\(c_i,c_j\)\]"
    r"|(?:d_\{ij\}|m_\{ijr\}|i)\s*(?:≥|≤|≠|=)\s*(?:128|0|1|j)"
    r"|\[[0-9]+,[0-9]+\)"
    r"|\[128,∞\)"
    r"|\([se],[se]\)"
    r"|p\(t\|s,e\)"
    r"|(?:LN|MLP|FFN|BCEWithLogits|Sigmoid|softmax|softplus|log|max|min)\(·\)"
    r"|(?:[A-Za-z]+|[σγλπρφακωφθηδβμντ])(?:'(?:_\{[^{}]*\}|_[A-Za-z0-9]+)?)?(?:(?:_\{[^{}]*\}|_[A-Za-z0-9]+)|(?:\^\{[^{}]*\}|\^[A-Za-z0-9]+))+"
    r"|[A-Za-z]+(?=\(·\))"
    r"|[σγλπρφακωφθηδβμντ]"
    r"|[⊙∈∋∑√≤≥≠]"
    r"|(?<![A-Za-z0-9])(?:BCEWithLogits|Sigmoid|softmax|softplus|LN|MLP|FFN|argmax|max|min|log|exp)(?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9])(?:L|q|p|U|r|w|u|y|G|H|m|V|v|k|e|T|d|c|a|I|B|z|E|f|Q|K|h|l|x|b|g|i|s|t|n|o|P|R)(?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9])E'(?![A-Za-z0-9])"
)


def make_run(text: str, roman: bool = False) -> etree._Element:
    run = etree.Element(M("r"))
    if roman:
        rpr = etree.SubElement(run, M("rPr"))
        style = etree.SubElement(rpr, M("sty"))
        style.set(M("val"), "p")
    t = etree.SubElement(run, M("t"))
    if text[:1].isspace() or text[-1:].isspace() or "  " in text:
        t.set(f"{{{XML_NS}}}space", "preserve")
    t.text = text
    return run


def consume_braced(src: str, pos: int) -> tuple[str, int]:
    if pos >= len(src) or src[pos] != "{":
        return "", pos
    depth = 0
    for i in range(pos, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[pos + 1 : i], i + 1
    return src[pos + 1 :], len(src)


def parse_math(src: str) -> list[etree._Element]:
    nodes: list[etree._Element] = []
    i = 0
    while i < len(src):
        start = i
        ch = src[i]
        if ch.isalpha() and ord(ch) < 128:
            i += 1
            while i < len(src) and src[i].isalpha() and ord(src[i]) < 128:
                i += 1
            base = src[start:i]
        elif ch in GREEK or ch in "∑√∏∫⊙":
            base = ch
            i += 1
        elif "\u3400" <= ch <= "\u9fff":
            i += 1
            while i < len(src) and "\u3400" <= src[i] <= "\u9fff":
                i += 1
            base = src[start:i]
        elif ch.isdigit():
            i += 1
            while i < len(src) and src[i].isdigit():
                i += 1
            nodes.append(make_run(src[start:i], roman=True))
            continue
        else:
            nodes.append(make_run(ch, roman=ch.isspace() or not ch.isalnum()))
            i += 1
            continue

        prime = False
        subs: list[str] = []
        sups: list[str] = []
        while i < len(src):
            if src[i] == "'":
                prime = True
                i += 1
            elif src[i] in "_^":
                kind = src[i]
                i += 1
                if i < len(src) and src[i] == "{":
                    value, i = consume_braced(src, i)
                elif i < len(src):
                    value = src[i]
                    i += 1
                else:
                    value = ""
                (subs if kind == "_" else sups).append(value)
            else:
                break
        if prime:
            sups.insert(0, "′")
        for sub in subs:
            # Preserve multiple suffixes without losing their order.
            if len(subs) > 1:
                sub = " ".join(subs)
                subs = [sub]
                break
        for sup in sups:
            if len(sups) > 1:
                sup = " ".join(sups)
                sups = [sup]
                break

        roman = base in FUNCTIONS or (len(base) > 1 and not all(ch in GREEK for ch in base)) or (
            base and "\u3400" <= base[0] <= "\u9fff"
        )
        if subs or sups:
            if subs and sups:
                node = etree.Element(M("sSubSup"))
                etree.SubElement(node, M("e")).extend([make_run(base, roman)])
                etree.SubElement(node, M("sub")).extend(parse_math(subs[0]))
                etree.SubElement(node, M("sup")).extend(parse_math(sups[0]))
            elif subs:
                node = etree.Element(M("sSub"))
                etree.SubElement(node, M("e")).append(make_run(base, roman))
                etree.SubElement(node, M("sub")).extend(parse_math(subs[0]))
            else:
                node = etree.Element(M("sSup"))
                etree.SubElement(node, M("e")).append(make_run(base, roman))
                etree.SubElement(node, M("sup")).extend(parse_math(sups[0]))
            nodes.append(node)
        else:
            nodes.append(make_run(base, roman))
    return nodes


def make_math_object(text: str) -> etree._Element:
    obj = etree.Element(M("oMath"))
    obj.extend(parse_math(text))
    return obj


def get_text(element: etree._Element) -> str:
    return "".join(element.xpath(".//w:t/text()", namespaces=NS))


def formula_paragraphs(root: etree._Element) -> list[etree._Element]:
    found = []
    for table in root.xpath(".//w:tbl", namespaces=NS):
        for row in table.xpath("./w:tr", namespaces=NS):
            cells = row.xpath("./w:tc", namespaces=NS)
            if len(cells) < 2:
                continue
            number = get_text(cells[1]).strip()
            if not re.fullmatch(r"\(\d+\)", number):
                continue
            paras = cells[0].xpath("./w:p", namespaces=NS)
            if paras and get_text(paras[0]).strip():
                found.append(paras[0])
    return found


def replace_formula_paragraph(paragraph: etree._Element) -> str:
    source = get_text(paragraph)
    ppr = paragraph.find(W("pPr"))
    for child in list(paragraph):
        if child is not ppr:
            paragraph.remove(child)
    math = make_math_object(source)
    if ppr is not None:
        paragraph.insert(1, math)
    else:
        paragraph.insert(0, math)
    return source


def make_text_run_like(original: etree._Element, text: str) -> etree._Element:
    run = etree.Element(W("r"))
    rpr = original.find(W("rPr"))
    if rpr is not None:
        run.append(copy.deepcopy(rpr))
    t = etree.SubElement(run, W("t"))
    if text[:1].isspace() or text[-1:].isspace() or "  " in text:
        t.set(f"{{{XML_NS}}}space", "preserve")
    t.text = text
    return run


def convert_inline_math(
    root: etree._Element, excluded_paragraphs: set[etree._Element]
) -> int:
    converted = 0
    for paragraph in root.xpath(".//w:p", namespaces=NS):
        if paragraph in excluded_paragraphs:
            continue
        for text_node in list(paragraph.xpath(".//w:t", namespaces=NS)):
            run = text_node.getparent()
            if run is None or run.tag != W("r"):
                continue
            source = text_node.text or ""
            matches = list(INLINE_PATTERN.finditer(source))
            if not matches:
                continue
            parent = run.getparent()
            if parent is None:
                continue
            replacement: list[etree._Element] = []
            cursor = 0
            for match in matches:
                if match.start() > cursor:
                    replacement.append(make_text_run_like(run, source[cursor : match.start()]))
                replacement.append(make_math_object(match.group()))
                converted += 1
                cursor = match.end()
            if cursor < len(source):
                replacement.append(make_text_run_like(run, source[cursor:]))
            index = parent.index(run)
            parent.remove(run)
            for offset, element in enumerate(replacement):
                parent.insert(index + offset, element)
    return converted


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        if digest(OUTPUT) != digest(SOURCE):
            raise FileExistsError(OUTPUT)
        OUTPUT.unlink()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE, OUTPUT)

    with ZipFile(OUTPUT, "r") as archive:
        package = {item.filename: archive.read(item.filename) for item in archive.infolist()}
    root = etree.fromstring(package[DOCUMENT_XML])
    formulas = formula_paragraphs(root)
    if len(formulas) != 44:
        raise RuntimeError(f"Expected 44 numbered formula rows, found {len(formulas)}")
    original_formula_texts = [replace_formula_paragraph(p) for p in formulas]
    excluded = set(formulas)
    inline_count = convert_inline_math(root, excluded)
    package[DOCUMENT_XML] = etree.tostring(
        root, encoding="UTF-8", xml_declaration=True, standalone=True
    )

    tmp = OUTPUT.with_suffix(".writing.docx")
    with ZipFile(tmp, "w", ZIP_DEFLATED) as archive:
        for name, data in package.items():
            archive.writestr(name, data)
    tmp.replace(OUTPUT)
    print(f"output={OUTPUT}")
    print(f"numbered_formula_objects={len(original_formula_texts)}")
    print(f"inline_math_objects={inline_count}")


if __name__ == "__main__":
    main()
