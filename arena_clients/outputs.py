"""Console summary and local copy of an image submission.

Both helpers run after the submission is final and must never change its
outcome: they never raise, never touch the network, and do no slow work.
An agent that crashed after submitting would exit non-zero, and an attempt
could look failed even though its answer counted.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

_IMAGE_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}


def missing_tools_note(result: Any) -> str:
    """One line naming required tools the run skipped, or "" if none.

    Practice sets quality to 0 and accepted to false when a required tool
    was not called, and lists them in ``score.missing_required_tools``.
    """
    try:
        score = result.get("score") if isinstance(result, dict) else getattr(result, "score", None)
        missing = score.get("missing_required_tools") if isinstance(score, dict) else None
        if isinstance(missing, (list, tuple)) and missing:
            return "Missing required tools: " + ", ".join(str(name) for name in missing)
        return ""
    except Exception:
        return ""


def submission_summary_lines(result: Any) -> list[str]:
    """Accepted / status / score lines for a submit result, without image data.

    Accepts the REST ``SubmitResult`` or the MCP ``arena.image.submit_edit``
    dict. Image submissions echo the image back (``edited_image`` and
    ``score.output_image_uri``), so the raw result must not be printed.
    """
    try:
        if isinstance(result, dict):
            accepted = result.get("accepted")
            status = result.get("status")
            score = result.get("score")
        else:
            accepted = getattr(result, "accepted", None)
            status = getattr(result, "status", None)
            score = getattr(result, "score", None)
        lines = [f"Accepted: {accepted}"]
        if status:
            lines.append(f"Status: {status}")
        if isinstance(score, dict) and score.get("final_score") is not None:
            lines.append(f"Score: {score['final_score']}")
        note = missing_tools_note(result)
        if note:
            lines.append(note)
        judge_error = score.get("judge_error") if isinstance(score, dict) else None
        if judge_error:
            # Quality is 0 because the judge failed, not because the image was poor.
            lines.append(f"Judge error: {judge_error} (quality 0 until the image can be judged)")
        return lines
    except Exception as exc:
        return [f"Submission result unavailable ({type(exc).__name__})"]


def save_submitted_image(agent_id: str, image_uri: Any, root: Path) -> str:
    """Best effort: write the submitted image to ``root/outputs/``.

    Only inline ``data:image/...;base64`` images are written; a URL is
    skipped rather than downloaded. Returns a one-line status to print.
    """
    try:
        header, _, data = str(image_uri or "").partition(",")
        if not header.startswith("data:image/") or ";base64" not in header:
            return "not saved (not an inline image)"
        mime = header[len("data:"):].split(";")[0]
        extension = _IMAGE_EXTENSIONS.get(mime, "img")
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(agent_id))[:64] or "agent"
        out_dir = Path(root) / "outputs"
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"{safe_id}-submission.{extension}"
        path.write_bytes(base64.b64decode(data, validate=True))
        return f"saved to {path}"
    except Exception as exc:
        return f"not saved ({type(exc).__name__})"
