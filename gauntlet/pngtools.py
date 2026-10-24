"""Stdlib image inspection for hidden data.

Covers the common CTF hiding spots without Pillow:
- format and dimensions (PNG, JPEG, GIF, WEBP, BMP)
- PNG text chunks (tEXt, zTXt, iTXt) and bytes appended after IEND
- JPEG comment segments and printable strings (EXIF text fields)
- least-significant-bit payloads in 8-bit PNG pixels
"""

from __future__ import annotations

import re
import struct
import zlib
from typing import Any

from .decoders import find_flags, english_score

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def sniff(data: bytes) -> str:
    if data.startswith(PNG_SIG):
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:2] == b"BM":
        return "bmp"
    return "unknown"


def dimensions(data: bytes) -> tuple[int, int] | None:
    kind = sniff(data)
    try:
        if kind == "png":
            return struct.unpack(">II", data[16:24])
        if kind == "gif":
            return struct.unpack("<HH", data[6:10])
        if kind == "bmp":
            w, h = struct.unpack("<ii", data[18:26])
            return w, abs(h)
        if kind == "jpeg":
            i = 2
            while i < len(data) - 9:
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xC0, 0xC1, 0xC2):
                    h, w = struct.unpack(">HH", data[i + 5 : i + 9])
                    return w, h
                seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
                i += 2 + seg_len
        if kind == "webp" and data[12:16] == b"VP8X":
            w = int.from_bytes(data[24:27], "little") + 1
            h = int.from_bytes(data[27:30], "little") + 1
            return w, h
    except struct.error:
        return None
    return None


def printable_strings(data: bytes, min_len: int = 6, limit: int = 40) -> list[str]:
    found = re.findall(rb"[\x20-\x7e]{%d,}" % min_len, data)
    out: list[str] = []
    for raw in found:
        text = raw.decode("ascii", errors="ignore")
        if english_score(text) > 0.45 or find_flags(text) or "=" in text or ":" in text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# PNG


def png_chunks(data: bytes) -> tuple[list[tuple[bytes, bytes]], bytes]:
    """Return ([(type, body), ...], trailing_bytes_after_IEND)."""
    chunks: list[tuple[bytes, bytes]] = []
    i = len(PNG_SIG)
    while i + 8 <= len(data):
        length = struct.unpack(">I", data[i : i + 4])[0]
        ctype = data[i + 4 : i + 8]
        body = data[i + 8 : i + 8 + length]
        chunks.append((ctype, body))
        i += 12 + length
        if ctype == b"IEND":
            return chunks, data[i:]
    return chunks, b""


def png_text(chunks: list[tuple[bytes, bytes]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for ctype, body in chunks:
        try:
            if ctype == b"tEXt":
                key, _, val = body.partition(b"\x00")
                out[key.decode("latin-1")] = val.decode("latin-1")
            elif ctype == b"zTXt":
                key, _, rest = body.partition(b"\x00")
                out[key.decode("latin-1")] = zlib.decompress(rest[1:]).decode("latin-1")
            elif ctype == b"iTXt":
                key, _, rest = body.partition(b"\x00")
                comp_flag = rest[0]
                rest = rest[2:]
                _lang, _, rest = rest.partition(b"\x00")
                _tkey, _, text = rest.partition(b"\x00")
                if comp_flag:
                    text = zlib.decompress(text)
                out[key.decode("latin-1")] = text.decode("utf-8", errors="replace")
        except Exception:
            continue
    return out


_CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


def png_pixels(data: bytes) -> tuple[int, int, int, bytes] | None:
    """Decode an 8-bit, non-interlaced PNG into raw rows. Returns (w, h, channels, bytes)."""
    chunks, _ = png_chunks(data)
    ihdr = next((b for t, b in chunks if t == b"IHDR"), None)
    if not ihdr:
        return None
    w, h, depth, ctype, _comp, _filt, interlace = struct.unpack(">IIBBBBB", ihdr)
    if depth != 8 or interlace != 0 or ctype not in _CHANNELS:
        return None
    if w * h > 4_000_000:
        return None
    ch = _CHANNELS[ctype]
    raw = zlib.decompress(b"".join(b for t, b in chunks if t == b"IDAT"))
    stride = w * ch
    out = bytearray(h * stride)
    prev = bytearray(stride)
    pos = 0
    for y in range(h):
        ftype = raw[pos]
        line = bytearray(raw[pos + 1 : pos + 1 + stride])
        pos += 1 + stride
        if ftype == 1:
            for x in range(ch, stride):
                line[x] = (line[x] + line[x - ch]) & 0xFF
        elif ftype == 2:
            for x in range(stride):
                line[x] = (line[x] + prev[x]) & 0xFF
        elif ftype == 3:
            for x in range(stride):
                left = line[x - ch] if x >= ch else 0
                line[x] = (line[x] + ((left + prev[x]) >> 1)) & 0xFF
        elif ftype == 4:
            for x in range(stride):
                a = line[x - ch] if x >= ch else 0
                b = prev[x]
                c = prev[x - ch] if x >= ch else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                line[x] = (line[x] + pred) & 0xFF
        out[y * stride : (y + 1) * stride] = line
        prev = line
    return w, h, ch, bytes(out)


def lsb_payloads(pixels: bytes, channels: int, max_bytes: int = 4096) -> list[dict[str, Any]]:
    """Try common LSB layouts: all color channels, each channel alone, MSB-first bits."""
    results: list[dict[str, Any]] = []
    color = min(channels, 3)
    layouts: list[tuple[str, list[int]]] = [("rgb" if color == 3 else "gray", list(range(color)))]
    if color == 3:
        layouts += [("r", [0]), ("g", [1]), ("b", [2])]
    for label, chans in layouts:
        bits: list[int] = []
        limit_bits = max_bytes * 8
        for i in range(0, len(pixels), channels):
            for c in chans:
                bits.append(pixels[i + c] & 1)
            if len(bits) >= limit_bits:
                break
        data = bytearray()
        for j in range(0, len(bits) - 7, 8):
            byte = 0
            for k in range(8):
                byte = (byte << 1) | bits[j + k]
            data.append(byte)
        text = _leading_text(bytes(data))
        if text and (english_score(text) > 0.5 or find_flags(text)):
            results.append({"layout": f"lsb-{label}", "text": text[:1000]})
    return results


def _leading_text(data: bytes) -> str:
    """Printable prefix of ``data``; also handles a 4-byte big-endian length header."""
    if len(data) > 4:
        n = struct.unpack(">I", data[:4])[0]
        if 0 < n <= len(data) - 4:
            body = data[4 : 4 + n]
            if all(32 <= b < 127 or b in (9, 10, 13) for b in body):
                return body.decode("ascii")
    out = bytearray()
    for b in data:
        if b == 0:
            break
        if not (32 <= b < 127 or b in (9, 10, 13)):
            break
        out.append(b)
    return out.decode("ascii") if len(out) >= 4 else ""


def inspect_image(data: bytes) -> dict[str, Any]:
    """One-shot summary of everything findable without a vision model."""
    report: dict[str, Any] = {"format": sniff(data), "bytes": len(data)}
    dims = dimensions(data)
    if dims:
        report["width"], report["height"] = dims
    findings: list[str] = []
    if report["format"] == "png":
        chunks, trailing = png_chunks(data)
        report["chunks"] = [t.decode("latin-1") for t, _ in chunks]
        text = png_text(chunks)
        if text:
            report["text_chunks"] = text
            findings += [f"{k}={v}" for k, v in text.items()]
        if trailing:
            report["trailing_bytes"] = len(trailing)
            tail_strings = printable_strings(trailing, min_len=4)
            if tail_strings:
                report["trailing_strings"] = tail_strings
                findings += tail_strings
        try:
            decoded = png_pixels(data)
        except Exception:
            decoded = None
        if decoded:
            _w, _h, ch, pixels = decoded
            lsb = lsb_payloads(pixels, ch)
            if lsb:
                report["lsb"] = lsb
                findings += [item["text"] for item in lsb]
    else:
        strings = printable_strings(data)
        if strings:
            report["strings"] = strings[:20]
            findings += strings[:20]
    flags = sorted({f for item in findings for f in find_flags(item)})
    if flags:
        report["flags"] = flags
    report["findings"] = findings[:30]
    return report


def make_png(width: int, height: int, rgb: bytes, text: dict[str, str] | None = None) -> bytes:
    """Tiny PNG writer (used by tests and the mock arena)."""
    def chunk(ctype: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(ctype + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + ctype + body + struct.pack(">I", crc)

    stride = width * 3
    raw = b"".join(b"\x00" + rgb[y * stride : (y + 1) * stride] for y in range(height))
    out = PNG_SIG + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    for key, val in (text or {}).items():
        out += chunk(b"tEXt", key.encode("latin-1") + b"\x00" + val.encode("latin-1"))
    out += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return out
