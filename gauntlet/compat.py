"""Read MCP objects the same way under mcp 1.x (camelCase) and 2.x (snake_case).

The kit pins mcp 1.29 for the real runtime. Local test machines may have 2.x.
Every field read in this package goes through these helpers.
"""

from __future__ import annotations

import base64
import json
from typing import Any


def attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Return the first present attribute or dict key from ``names``."""
    if obj is None:
        return default
    for name in names:
        if isinstance(obj, dict):
            if name in obj and obj[name] is not None:
                return obj[name]
            continue
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def tool_name(tool: Any) -> str:
    return str(attr(tool, "name", default="") or "").strip()


def tool_description(tool: Any) -> str:
    return str(attr(tool, "description", default="") or "").strip()


def tool_schema(tool: Any) -> dict[str, Any]:
    schema = attr(tool, "inputSchema", "input_schema", default=None)
    if hasattr(schema, "model_dump"):
        schema = schema.model_dump()
    return schema if isinstance(schema, dict) else {}


def tool_annotations(tool: Any) -> dict[str, Any]:
    ann = attr(tool, "annotations", default=None)
    if ann is None:
        return {}
    if hasattr(ann, "model_dump"):
        try:
            return {k: v for k, v in ann.model_dump().items() if v is not None}
        except Exception:
            return {}
    return ann if isinstance(ann, dict) else {}


def result_is_error(result: Any) -> bool:
    return bool(attr(result, "isError", "is_error", default=False))


def result_structured(result: Any) -> Any:
    return attr(result, "structuredContent", "structured_content", default=None)


def content_blocks(result: Any) -> list[Any]:
    blocks = attr(result, "content", default=None)
    return list(blocks) if isinstance(blocks, (list, tuple)) else []


def parse_tool_result(result: Any) -> dict[str, Any]:
    """Flatten a CallToolResult into a plain dict without losing blocks.

    Returns keys:
      data:   parsed JSON (dict/list) when the first text block or the
              structured content is JSON, else None
      text:   all text blocks joined
      images: list of {"mime": str, "b64": str}
      resources: list of embedded resource dicts {"uri", "text", "b64", "mime"}
      is_error: bool
    """
    out: dict[str, Any] = {
        "data": None,
        "text": "",
        "images": [],
        "resources": [],
        "is_error": result_is_error(result),
    }
    structured = result_structured(result)
    texts: list[str] = []
    for block in content_blocks(result):
        btype = str(attr(block, "type", default="") or "")
        if btype == "text":
            texts.append(str(attr(block, "text", default="") or ""))
        elif btype in {"image", "audio"}:
            out["images"].append(
                {
                    "mime": str(attr(block, "mimeType", "mime_type", default="") or ""),
                    "b64": str(attr(block, "data", default="") or ""),
                }
            )
        elif btype == "resource":
            res = attr(block, "resource", default=None)
            out["resources"].append(
                {
                    "uri": str(attr(res, "uri", default="") or ""),
                    "text": attr(res, "text", default=None),
                    "b64": attr(res, "blob", default=None),
                    "mime": str(attr(res, "mimeType", "mime_type", default="") or ""),
                }
            )
        elif btype == "resource_link":
            out["resources"].append(
                {"uri": str(attr(block, "uri", default="") or ""), "text": None, "b64": None, "mime": ""}
            )
        else:
            maybe = attr(block, "text", default=None)
            if isinstance(maybe, str):
                texts.append(maybe)
    out["text"] = "\n".join(t for t in texts if t)
    if isinstance(structured, (dict, list)):
        out["data"] = structured
    elif texts:
        try:
            out["data"] = json.loads(texts[0])
        except (json.JSONDecodeError, TypeError):
            out["data"] = None
    return out


def result_as_dict(parsed: dict[str, Any]) -> dict[str, Any]:
    """The kit's convention: a dict, with raw text under "raw" when not JSON."""
    data = parsed.get("data")
    if isinstance(data, dict):
        return dict(data)
    if data is not None:
        return {"structured": data}
    if parsed.get("text"):
        return {"raw": parsed["text"]}
    return {}


def decode_b64_loose(value: str) -> bytes | None:
    """Decode base64 or a data: URI, tolerating missing padding."""
    if not isinstance(value, str) or not value:
        return None
    payload = value
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    payload = "".join(payload.split())
    payload += "=" * (-len(payload) % 4)
    try:
        return base64.b64decode(payload, validate=False)
    except Exception:
        try:
            return base64.urlsafe_b64decode(payload)
        except Exception:
            return None


def short(value: Any, limit: int = 600) -> str:
    """Compact JSON-ish rendering for prompts and logs, with long base64 elided."""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(_elide(value), ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
    if len(text) > limit:
        return text[: limit - 20] + f"...[+{len(text) - limit + 20} chars]"
    return text


def _elide(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _elide(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_elide(v) for v in value]
    if isinstance(value, str) and len(value) > 400 and (
        value.startswith("data:") or _mostly_b64(value)
    ):
        return f"<binary {len(value)} chars>"
    return value


def _mostly_b64(value: str) -> bool:
    sample = value[:500]
    ok = sum(c.isalnum() or c in "+/=_-" for c in sample)
    return ok / max(1, len(sample)) > 0.97 and " " not in sample
