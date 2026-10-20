"""
极简 PDF 文本提取（无第三方依赖），用于读取本地 arXiv 论文。

只处理最常见的情况：FlateDecode 内容流 + Tj/TJ 文本操作符。
TJ 数组里的字距数字必须剔除，否则文本会被数字污染（这是上一次提取失败的原因）。
"""

import re
import sys
import zlib

ESCAPES = {b'n': b'\n', b'r': b'\r', b't': b'\t', b'b': b'\b',
           b'f': b'\f', b'(': b'(', b')': b')', b'\\': b'\\'}


def _unescape(raw: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(raw):
        c = raw[i:i + 1]
        if c == b'\\' and i + 1 < len(raw):
            nxt = raw[i + 1:i + 2]
            if nxt in ESCAPES:
                out += ESCAPES[nxt]
                i += 2
                continue
            # 八进制转义
            m = re.match(rb'[0-7]{1,3}', raw[i + 1:i + 4])
            if m:
                out.append(int(m.group(0), 8) & 0xFF)
                i += 1 + len(m.group(0))
                continue
            i += 2
            continue
        out.append(raw[i])
        i += 1
    return bytes(out)


_STR_B = rb'\((?:[^()\\]|\\.)*\)'
_STR_S = r'\((?:[^()\\]|\\.)*\)'


def _content_to_text(s: str) -> str:
    """把内容流还原成可读文本（保持换行）。"""
    # 1) TJ 数组：只保留括号里的字符串，丢掉字距数字
    def tj(m):
        parts = re.findall(_STR_S, m.group(1))
        return ''.join(p[1:-1] for p in parts)

    s = re.sub(r'\[((?:[^\[\]\\]|\\.)*)\]\s*TJ', tj, s)

    # 2) 单个 Tj
    def tja(m):
        return m.group(0)[1:m.group(0).rfind(')')]

    s = re.sub(_STR_S + r'\s*Tj', tja, s)

    # 3) 换行/定位操作符 -> 换行符
    s = re.sub(r'\b(?:Td|TD|T\*|ET)\b', '\n', s)

    # 4) 去掉剩余的 PDF 操作符（保留换行）
    s = re.sub(r'\n{3,}', '\n\n', s)
    s = re.sub(r'[^\n\x20-\x7e\u00a0-\uffff]', '', s)
    s = re.sub(r'[ \t]{2,}', ' ', s)
    return s


def extract(pdf_path: str) -> str:
    raw = open(pdf_path, 'rb').read()
    streams = []
    for m in re.finditer(rb'stream\r?\n', raw):
        start = m.end()
        end = raw.find(b'endstream', start)
        if end < 0:
            continue
        blob = raw[start:end]
        try:
            streams.append(zlib.decompress(blob))
        except Exception:                             # noqa: BLE001
            continue

    pages = []
    for blob in streams:
        if b'Tj' not in blob and b'TJ' not in blob:
            continue
        # 先按字节反转义，再按 latin-1 解码（保持单字节映射）
        s = _unescape(blob).decode('latin-1')
        pages.append(_content_to_text(s))
    return '\n\n===== PAGE BREAK =====\n\n'.join(pages)


if __name__ == '__main__':
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else '_pdf_text.txt'
    text = extract(src)
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'wrote {dst}: {len(text)} chars')
