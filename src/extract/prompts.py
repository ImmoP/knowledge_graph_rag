"""Prompt-template loading and rendering (#23).

Implements ``docs/specs/extraction-contract.md`` §5 verbatim.

* Both physical prompt files live under ``prompts/`` at the repo root
  and are read from disk at run time — never inlined in Python.
* Startup verifies both files exist BEFORE any LLM call; missing
  prompts are a fatal startup error naming the exact missing path.
* Templates are rendered at run time from the ontology's
  per-type allowed-name lists and few-shot examples so an ontology
  edit is the only change needed to keep prompts honest.
* The template filenames and exact-content SHA256 are returned so the
  caller can record them in the run manifest (spec §3.1 sibling
  manifest) and mix them into the cache identity (spec §10.1).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "prompts"

ENTITY_PROMPT_FILE = "extract_entities_v1.txt"
RELATION_PROMPT_FILE = "extract_relations_v1.txt"

ENTITY_PROMPT_PATH = PROMPTS_DIR / ENTITY_PROMPT_FILE
RELATION_PROMPT_PATH = PROMPTS_DIR / RELATION_PROMPT_FILE


class PromptArtifactMissing(RuntimeError):
    """Raised at startup when a required prompt artifact is absent.

    Per spec §5.1: "Missing prompt => fatal startup error naming the
    exact missing path."
    """


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_prompt_artifacts(
    prompts_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Read both prompt files from disk; fail closed if any is missing.

    Returns a per-template dict with ``path``, ``sha256``, ``bytes``,
    and ``text`` keys. Both files are read fully into memory because
    their total size is small (a few KB) and the rendering code needs
    the text.

    Spec §5.1: "Startup must verify both files exist BEFORE any LLM
    call. Missing prompt => fatal startup error naming the exact
    missing path."
    """
    base = Path(prompts_dir) if prompts_dir is not None else PROMPTS_DIR
    out: dict[str, dict[str, Any]] = {}
    for name, path in (
        (ENTITY_PROMPT_FILE, base / ENTITY_PROMPT_FILE),
        (RELATION_PROMPT_FILE, base / RELATION_PROMPT_FILE),
    ):
        if not path.is_file():
            raise PromptArtifactMissing(
                f"required prompt artifact missing: {path}"
            )
        raw = path.read_bytes()
        out[name] = {
            "path": path,
            "filename": name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": raw,
            "text": raw.decode("utf-8"),
        }
    return out


def render_entity_prompt(
    *,
    template: str,
    allowed_entity_types: list[str],
    entity_type_definitions: str,
    verified_examples: str,
    chunk_text: str,
) -> str:
    """Substitute the entity-prompt placeholders.

    The template's placeholders are the literal ``{name}`` tokens
    defined in the physical artifact; substitution is positional per
    name so the template author can rearrange the prose without
    breaking the renderer.
    """
    allowed = ", ".join(allowed_entity_types) if allowed_entity_types else "(none)"
    return template.format(
        allowed_entity_types=allowed,
        entity_type_definitions=entity_type_definitions,
        verified_examples=verified_examples,
        chunk_text=chunk_text,
    )


def render_relation_prompt(
    *,
    template: str,
    allowed_relation_types: list[str],
    relation_definitions: str,
    chunk_id: str,
    doc_id: str,
    mention_list: str,
    chunk_text: str,
) -> str:
    """Substitute the relation-prompt placeholders."""
    allowed = ", ".join(allowed_relation_types) if allowed_relation_types else "(none)"
    return template.format(
        allowed_relation_types=allowed,
        relation_definitions=relation_definitions,
        chunk_id=chunk_id,
        doc_id=doc_id,
        mention_list=mention_list,
        chunk_text=chunk_text,
    )


def render_mention_list(mentions: list[dict]) -> str:
    """Render the validated mention surface for the relation prompt.

    Per spec §5.3, the relation prompt receives "the validated
    same-chunk mention records after their m_* IDs have been
    assigned." Each line gives ``mention_id``, ``entity_type``,
    ``surface_form``, and the char range. The pipeline never prints
    ``canonical_form`` to this surface because the relation prompt's
    only consumer is the model's matching of ``kind: "mention"``
    references to supplied IDs.
    """
    if not mentions:
        return "(no same-chunk mentions available)"
    lines: list[str] = []
    for m in mentions:
        lines.append(
            "- {mid}  entity_type={et}  surface_form={sf!r}  "
            "char_range=[{cs},{ce})".format(
                mid=m["mention_id"],
                et=m["entity_type"],
                sf=m["surface_form"],
                cs=m["evidence"]["char_start"],
                ce=m["evidence"]["char_end"],
            )
        )
    return "\n".join(lines)