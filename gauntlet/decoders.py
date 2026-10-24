"""Stdlib decoders for encoded or lightly encrypted challenge data.

The black box promises "encrypted data". In CTF-style events that usually
means layered encodings (base64 of hex of rot13), classical ciphers (Caesar,
Vigenere, Atbash, rail fence), XOR with a short key, or compressed blobs.
This module tries those layers automatically and ranks the results by how
much they look like English or like a flag.

Nothing here needs a third-party package, so the kit's lockfile stays as is.
"""

from __future__ import annotations

import base64
import binascii
import bz2
import codecs
import gzip
import html
import json
import lzma
import re
import string
import zlib
from dataclasses import dataclass, field
from typing import Callable, Iterable
from urllib.parse import unquote, unquote_plus

FLAG_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]{1,24}\{[^{}\s]{2,160}\}")
# Prefixes that make a WORD{...} match count as a real flag. A rotated flag
# such as SYNT{...} still matches FLAG_RE, so the prefix check keeps the
# search from stopping on ciphertext that only looks like a flag.
FLAG_PREFIXES = {
    "flag", "ctf", "nvidia", "nv", "key", "gauntlet", "arena", "secret",
    "answer", "token", "code", "pass", "password", "agent", "gtc", "dli",
}
_BYTE_DECODERS = {"hex", "base64", "base64url", "base32", "base85", "binary"}

COMMON_WORDS = (
    "the", "and", "you", "that", "for", "with", "this", "flag", "key", "secret",
    "password", "answer", "code", "tool", "call", "use", "next", "find", "hidden",
    "unlock", "is", "of", "to", "in", "it", "a", "are", "your", "from", "token",
    "clue", "vault", "door", "open", "name", "submit", "agent", "message", "data",
)
_WORD_RE = re.compile(r"[a-z]{2,}")
_SUBSTRING_WORDS = (
    "the", "and", "flag", "key", "secret", "password", "answer", "code", "tool",
    "hidden", "unlock", "vault", "open", "are", "discover", "once", "flee", "attack",
    "dawn", "meet", "north", "south", "east", "west", "agent", "door", "call", "use",
)
_PRINTABLE = set(string.printable)

MORSE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F",
    "--.": "G", "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L",
    "--": "M", "-.": "N", "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R",
    "...": "S", "-": "T", "..-": "U", "...-": "V", ".--": "W", "-..-": "X",
    "-.--": "Y", "--..": "Z", "-----": "0", ".----": "1", "..---": "2",
    "...--": "3", "....-": "4", ".....": "5", "-....": "6", "--...": "7",
    "---..": "8", "----.": "9", ".-.-.-": ".", "--..--": ",", "..--..": "?",
    "-..-.": "/", "-....-": "-", "-.--.": "(", "-.--.-": ")", "..--.-": "_",
    "-.-.--": "!", "---...": ":", ".-..-.": '"', ".----.": "'", "-...-": "=",
}


@dataclass
class Candidate:
    chain: list[str]
    text: str
    score: float
    flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "chain": " -> ".join(self.chain) or "identity",
            "text": self.text[:2000],
            "score": round(self.score, 3),
            "flags": self.flags,
        }


# ---------------------------------------------------------------------------
# Scoring


def english_score(text: str) -> float:
    """0..1 score: printable, letter-heavy, and containing common words."""
    if not text:
        return 0.0
    n = len(text)
    printable = sum(ch in _PRINTABLE for ch in text) / n
    if printable < 0.85:
        return printable * 0.2
    letters = sum(ch.isalpha() or ch == " " for ch in text) / n
    words = _WORD_RE.findall(text.lower())
    hits = sum(1 for w in words if w in COMMON_WORDS)
    word_ratio = hits / max(1, len(words))
    if text.count(" ") <= 1 and n >= 10:
        # Spaceless text (rail fence, packed ciphertext): look for words inside.
        low = text.lower()
        covered = sum(len(w) * low.count(w) for w in _SUBSTRING_WORDS)
        word_ratio = max(word_ratio, min(1.0, covered / n * 1.2))
    spaces = text.count(" ") / n
    space_bonus = 0.15 if 0.08 <= spaces <= 0.3 else 0.0
    vowels = sum(ch in "aeiouAEIOU" for ch in text) / max(1, sum(ch.isalpha() for ch in text))
    vowel_bonus = 0.1 if 0.25 <= vowels <= 0.5 else 0.0
    score = 0.35 * letters + 0.4 * min(1.0, word_ratio * 2.5) + space_bonus + vowel_bonus
    if find_flags(text):
        score += 0.6
    return min(1.5, score)


def find_flags(text: str) -> list[str]:
    """Flag-shaped matches whose prefix is a known flag word."""
    out = set()
    for match in FLAG_RE.findall(text or ""):
        prefix = match.split("{", 1)[0].lower()
        if prefix in FLAG_PREFIXES or prefix.endswith("flag") or prefix.endswith("ctf"):
            out.add(match)
    return sorted(out)


def find_flag_shapes(text: str) -> list[str]:
    """Every WORD{...} match, including unknown prefixes. For hints only."""
    return sorted(set(FLAG_RE.findall(text or "")))


# ---------------------------------------------------------------------------
# Byte helpers


def _to_text(data: bytes) -> str | None:
    for enc in ("utf-8", "latin-1"):
        try:
            text = data.decode(enc)
        except UnicodeDecodeError:
            continue
        if not text:
            return None
        printable = sum(ch in _PRINTABLE for ch in text) / len(text)
        if printable >= 0.85 or enc == "utf-8":
            return text
    return None


def _maybe_decompress(data: bytes) -> bytes | None:
    for fn in (
        lambda d: gzip.decompress(d),
        lambda d: zlib.decompress(d),
        lambda d: zlib.decompress(d, -15),
        lambda d: bz2.decompress(d),
        lambda d: lzma.decompress(d),
    ):
        try:
            out = fn(data)
            if out:
                return out
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Transforms. Each returns a decoded string or None when it does not apply.

_B64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")
_B64URL_RE = re.compile(r"^[A-Za-z0-9\-_=\s]+$")
_B32_RE = re.compile(r"^[A-Z2-7=\s]+$")
_HEX_RE = re.compile(r"^(?:0x)?[0-9a-fA-F\s:]+$")
_BIN_RE = re.compile(r"^[01\s]+$")
_OCT_RE = re.compile(r"^[0-7]{3}(?:[\s,][0-7]{3})*$")
_DEC_RE = re.compile(r"^\d{1,3}(?:[\s,]+\d{1,3})+$")
_MORSE_RE = re.compile(r"^[.\-/\s|]+$")


def t_base64(s: str) -> str | None:
    raw = "".join(s.split())
    if len(raw) < 8 or not _B64_RE.match(raw):
        return None
    try:
        data = base64.b64decode(raw + "=" * (-len(raw) % 4), validate=True)
    except (binascii.Error, ValueError):
        return None
    return _bytes_result(data)


def t_base64url(s: str) -> str | None:
    raw = "".join(s.split())
    if len(raw) < 8 or not _B64URL_RE.match(raw) or not re.search(r"[-_]", raw):
        return None
    try:
        data = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
    except (binascii.Error, ValueError):
        return None
    return _bytes_result(data)


def t_base32(s: str) -> str | None:
    raw = "".join(s.split()).upper()
    if len(raw) < 8 or not _B32_RE.match(raw):
        return None
    try:
        data = base64.b32decode(raw + "=" * (-len(raw) % 8))
    except (binascii.Error, ValueError):
        return None
    return _bytes_result(data)


def t_base85(s: str) -> str | None:
    raw = s.strip()
    if len(raw) < 10:
        return None
    for fn in (base64.b85decode, base64.a85decode):
        try:
            data = fn(raw.removeprefix("<~").removesuffix("~>"))
        except Exception:
            continue
        out = _bytes_result(data)
        if out and english_score(out) > 0.45:
            return out
    return None


def t_hex(s: str) -> str | None:
    raw = re.sub(r"[\s:]", "", s.strip())
    raw = raw[2:] if raw.lower().startswith("0x") else raw
    if len(raw) < 4 or len(raw) % 2 or not _HEX_RE.match(raw):
        return None
    try:
        return _bytes_result(bytes.fromhex(raw))
    except ValueError:
        return None


def t_binary(s: str) -> str | None:
    if not _BIN_RE.match(s.strip()):
        return None
    bits = "".join(s.split())
    if len(bits) < 8:
        return None
    groups = s.split()
    if all(len(g) in (7, 8) for g in groups) and len(groups) > 1:
        data = bytes(int(g, 2) for g in groups)
    elif len(bits) % 8 == 0:
        data = bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8))
    else:
        return None
    return _bytes_result(data)


def t_octal(s: str) -> str | None:
    if not _OCT_RE.match(s.strip()):
        return None
    try:
        return "".join(chr(int(x, 8)) for x in re.split(r"[\s,]+", s.strip()))
    except ValueError:
        return None


def t_decimal(s: str) -> str | None:
    if not _DEC_RE.match(s.strip()):
        return None
    nums = [int(x) for x in re.split(r"[\s,]+", s.strip()) if x]
    if all(32 <= n <= 126 for n in nums):
        return "".join(chr(n) for n in nums)
    if all(1 <= n <= 26 for n in nums):
        return "".join(chr(ord("A") + n - 1) for n in nums)
    return None


def t_rot13(s: str) -> str | None:
    if not re.search(r"[A-Za-z]", s):
        return None
    return codecs.encode(s, "rot_13")


def t_rot47(s: str) -> str | None:
    if not s or not any(33 <= ord(c) <= 126 for c in s):
        return None
    return "".join(
        chr(33 + ((ord(c) - 33 + 47) % 94)) if 33 <= ord(c) <= 126 else c for c in s
    )


def caesar(s: str, shift: int) -> str:
    out = []
    for ch in s:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - 97 + shift) % 26 + 97))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - 65 + shift) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)


def t_caesar_best(s: str) -> str | None:
    if sum(ch.isalpha() for ch in s) < 4:
        return None
    best = max(range(1, 26), key=lambda k: english_score(caesar(s, -k)))
    return caesar(s, -best)


def t_atbash(s: str) -> str | None:
    if not re.search(r"[A-Za-z]", s):
        return None
    out = []
    for ch in s:
        if "a" <= ch <= "z":
            out.append(chr(219 - ord(ch)))
        elif "A" <= ch <= "Z":
            out.append(chr(155 - ord(ch)))
        else:
            out.append(ch)
    return "".join(out)


def t_reverse(s: str) -> str | None:
    return s[::-1] if len(s) > 3 else None


def t_morse(s: str) -> str | None:
    raw = s.strip()
    if not _MORSE_RE.match(raw) or not re.search(r"[.\-]", raw):
        return None
    words = re.split(r"\s*[/|]\s*|\s{3,}", raw)
    out_words = []
    for word in words:
        letters = [MORSE.get(code, "?") for code in word.split()]
        out_words.append("".join(letters))
    text = " ".join(w for w in out_words if w)
    return text if text and text.count("?") <= len(text) // 4 else None


def t_url(s: str) -> str | None:
    if "%" not in s and "+" not in s:
        return None
    out = unquote_plus(s) if "+" in s and "%" in s else unquote(s)
    return out if out != s else None


def t_html(s: str) -> str | None:
    if "&" not in s:
        return None
    out = html.unescape(s)
    return out if out != s else None


def t_unicode_escape(s: str) -> str | None:
    if "\\u" not in s and "\\x" not in s:
        return None
    try:
        return codecs.decode(s, "unicode_escape")
    except Exception:
        return None


def t_rail_fence_best(s: str) -> str | None:
    if len(s) < 8 or " " in s.strip() and len(s.split()) > 3:
        return None
    best_text, best_score = None, english_score(s)
    for rails in range(2, 7):
        text = rail_fence_decrypt(s, rails)
        sc = english_score(text)
        if sc > best_score + 0.15:
            best_text, best_score = text, sc
    return best_text


def t_jwt(s: str) -> str | None:
    parts = s.strip().split(".")
    if len(parts) != 3 or not all(parts[:2]):
        return None
    decoded = []
    for part in parts[:2]:
        try:
            decoded.append(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)).decode("utf-8"))
        except Exception:
            return None
    return f"header={decoded[0]} payload={decoded[1]}"


def t_xor_single_byte(s: str) -> str | None:
    """Only meaningful on binary-looking text (latin-1 bytes from a prior decode)."""
    data = s.encode("latin-1", errors="ignore")
    if len(data) < 6:
        return None
    if english_score(s) > 0.6:
        return None
    best, best_sc = None, 0.0
    for k in range(1, 256):
        cand = bytes(b ^ k for b in data)
        text = cand.decode("latin-1")
        sc = english_score(text)
        if sc > best_sc:
            best, best_sc = text, sc
    return best if best_sc > 0.5 else None


def _bytes_result(data: bytes) -> str | None:
    if not data:
        return None
    inflated = _maybe_decompress(data)
    if inflated:
        data = inflated
    return _to_text(data)


TRANSFORMS: dict[str, Callable[[str], str | None]] = {
    "base64": t_base64,
    "base64url": t_base64url,
    "base32": t_base32,
    "base85": t_base85,
    "hex": t_hex,
    "binary": t_binary,
    "octal": t_octal,
    "decimal": t_decimal,
    "morse": t_morse,
    "url": t_url,
    "html": t_html,
    "unicode_escape": t_unicode_escape,
    "jwt": t_jwt,
    "rot13": t_rot13,
    "caesar": t_caesar_best,
    "atbash": t_atbash,
    "rot47": t_rot47,
    "reverse": t_reverse,
    "rail_fence": t_rail_fence_best,
    "xor1": t_xor_single_byte,
}

# Transforms that always produce output; only keep them when they improve the score.
_SPECULATIVE = {"rot13", "caesar", "atbash", "rot47", "reverse", "rail_fence", "xor1"}


# ---------------------------------------------------------------------------
# Keyed ciphers (key supplied by the caller, usually found elsewhere in the box)


def vigenere_decrypt(text: str, key: str) -> str:
    key_shifts = [ord(c.lower()) - 97 for c in key if c.isalpha()]
    if not key_shifts:
        return text
    out, i = [], 0
    for ch in text:
        if ch.isalpha():
            shift = key_shifts[i % len(key_shifts)]
            base = 65 if ch.isupper() else 97
            out.append(chr((ord(ch) - base - shift) % 26 + base))
            i += 1
        else:
            out.append(ch)
    return "".join(out)


def vigenere_encrypt(text: str, key: str) -> str:
    inverse = "".join(chr((26 - (ord(c.lower()) - 97)) % 26 + 97) for c in key if c.isalpha())
    return vigenere_decrypt(text, inverse)


def xor_with_key(data: bytes, key: bytes) -> bytes:
    if not key:
        return data
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


def rail_fence_decrypt(text: str, rails: int) -> str:
    if rails < 2 or rails >= len(text):
        return text
    pattern = list(range(rails)) + list(range(rails - 2, 0, -1))
    order = sorted(range(len(text)), key=lambda i: (pattern[i % len(pattern)], i))
    out = [""] * len(text)
    for ch, idx in zip(text, order):
        out[idx] = ch
    return "".join(out)


def keyed_attempts(blob: str, keys: Iterable[str]) -> list[Candidate]:
    """Try every key with Vigenere and repeating-key XOR (on raw or base64/hex bytes)."""
    results: list[Candidate] = []
    raw_variants: list[tuple[str, bytes]] = [("raw", blob.encode("latin-1", errors="ignore"))]
    for name in ("base64", "hex"):
        try:
            if name == "base64":
                clean = "".join(blob.split())
                data = base64.b64decode(clean + "=" * (-len(clean) % 4), validate=True)
            else:
                data = bytes.fromhex(re.sub(r"\s", "", blob))
            raw_variants.append((name, data))
        except Exception:
            continue
    for key in keys:
        key = str(key or "").strip()
        if not key or len(key) > 64:
            continue
        text = vigenere_decrypt(blob, key)
        results.append(Candidate([f"vigenere({key})"], text, english_score(text), find_flags(text)))
        for label, data in raw_variants:
            out = xor_with_key(data, key.encode())
            text = out.decode("latin-1")
            chain = [label, f"xor({key})"] if label != "raw" else [f"xor({key})"]
            results.append(Candidate(chain, text, english_score(text), find_flags(text)))
    results.sort(key=lambda c: c.score, reverse=True)
    return results[:5]


# ---------------------------------------------------------------------------
# Search


def auto_decode(
    text: str,
    *,
    max_depth: int = 4,
    beam: int = 10,
    keys: Iterable[str] = (),
) -> list[Candidate]:
    """Beam search over transform chains. Returns candidates, best first.

    A chain stops growing once its output scores like English or holds a flag.
    """
    text = (text or "").strip()
    if not text:
        return []
    start = Candidate([], text, english_score(text), find_flags(text))
    seen = {text}
    frontier = [start]
    found: list[Candidate] = [start]
    for _ in range(max_depth):
        nxt: list[Candidate] = []
        for cand in frontier:
            if cand.flags or (cand.chain and cand.score >= 0.9):
                continue
            for name, fn in TRANSFORMS.items():
                if cand.chain and cand.chain[-1] == name and name in {"rot13", "atbash", "reverse", "rot47"}:
                    continue
                if name == "xor1" and not (cand.chain and cand.chain[-1] in _BYTE_DECODERS):
                    continue
                try:
                    out = fn(cand.text)
                except Exception:
                    out = None
                if not out or out in seen or len(out) > 20000:
                    continue
                score = english_score(out)
                if name in _SPECULATIVE and score <= cand.score + 0.1:
                    continue
                seen.add(out)
                child = Candidate(cand.chain + [name], out, score, find_flags(out))
                nxt.append(child)
        if not nxt:
            break
        nxt.sort(key=lambda c: (bool(c.flags), c.score), reverse=True)
        frontier = nxt[:beam]
        found.extend(frontier)
    keyed = keyed_attempts(text, keys) if keys else []
    found.extend(keyed)
    found.sort(key=lambda c: (bool(c.flags), c.score, -len(c.chain)), reverse=True)
    unique: list[Candidate] = []
    texts: set[str] = set()
    for c in found:
        if c.text in texts:
            continue
        texts.add(c.text)
        unique.append(c)
    return unique[:8]


_SPAN_PATTERNS = (
    ("base64", re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{16,}={0,2}(?![A-Za-z0-9+/=])")),
    ("hex", re.compile(r"\b(?:0x)?(?:[0-9a-fA-F]{2}){8,}\b")),
    ("binary", re.compile(r"\b(?:[01]{8}\s+){3,}[01]{8}\b")),
    ("morse", re.compile(r"(?:[.\-]{1,6}[ /]+){4,}[.\-]{1,6}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]*")),
    ("decimal", re.compile(r"\b(?:\d{2,3}[ ,]+){5,}\d{2,3}\b")),
)


def find_encoded_spans(text: str, limit: int = 12) -> list[tuple[str, str]]:
    """Locate substrings that look encoded inside otherwise normal prose."""
    spans: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, pattern in _SPAN_PATTERNS:
        for match in pattern.finditer(text or ""):
            span = match.group(0).strip()
            if span in seen:
                continue
            if label == "base64" and (span.isalpha() and span.islower()):
                continue
            seen.add(span)
            spans.append((label, span))
            if len(spans) >= limit:
                return spans
    return spans


def decode_findings(text: str, *, keys: Iterable[str] = (), min_gain: float = 0.25) -> list[dict]:
    """Decode the encoded-looking spans inside ``text``. Keeps only real gains."""
    findings: list[dict] = []
    keys = list(keys)
    candidates = find_encoded_spans(text)
    whole = (text or "").strip()
    if whole and len(whole) <= 4000 and english_score(whole) < 0.45:
        candidates.insert(0, ("whole", whole))
    for label, span in candidates:
        results = auto_decode(span, keys=keys)
        if not results:
            continue
        base = english_score(span)
        best = results[0]
        if not best.chain:
            if len(results) < 2:
                continue
            best = results[1]
        if best.flags or best.score >= base + min_gain:
            findings.append({"source": label, "input": span[:200], **best.as_dict()})
    return findings


def describe_transforms() -> str:
    return ", ".join(TRANSFORMS) + ", vigenere(key), xor(key), rail_fence(n), caesar(n)"


def apply_named(text: str, op: str, key: str = "") -> str:
    """Apply one named transform, used by the LLM-facing decode tool."""
    op = (op or "").strip().lower()
    if op == "vigenere":
        return vigenere_decrypt(text, key)
    if op == "vigenere_encrypt":
        return vigenere_encrypt(text, key)
    if op == "xor":
        data = text.encode("latin-1", errors="ignore")
        return xor_with_key(data, key.encode()).decode("latin-1")
    if op == "caesar":
        try:
            return caesar(text, -int(key or "3"))
        except ValueError:
            return t_caesar_best(text) or text
    if op == "rail_fence":
        try:
            return rail_fence_decrypt(text, int(key or "3"))
        except ValueError:
            return text
    fn = TRANSFORMS.get(op)
    if fn is None:
        raise ValueError(f"unknown op {op!r}; known: {describe_transforms()}")
    out = fn(text)
    if out is None:
        raise ValueError(f"{op} does not apply to this input")
    return out


def to_json_safe(cands: list[Candidate]) -> str:
    return json.dumps([c.as_dict() for c in cands], ensure_ascii=False)
