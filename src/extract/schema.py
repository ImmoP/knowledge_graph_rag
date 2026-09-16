"""Ontology-driven JSON Schema generation (#23).

Renders the JSON Schemas for the two extraction-phase outputs from
``ontology.yaml`` filtered to ``population_mode == extracted``. Per
spec §5.2 / §5.3 / §8.2, no Python constant may reproduce the v1
vocabulary — the schemas are built directly from the YAML every run.

The schema is passed verbatim to :meth:`LLMProvider.generate_json`,
which owns its one schema-validation retry per spec §2.2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


#: JSON Schema versions emitted for the manifest's ``schema_version``
#: field. The implementing ticket bumps these on any schema shape
#: change; both schemas share a single version because they are
#: generated together from the same ontology.
SCHEMA_VERSION = 1

#: Schema version for the v2 extraction contract (#51, #47). The v2
#: model-facing schemas do not require the model to author char
#: offsets; the deterministic exact-span v2 aligner (#49) derives
#: offsets from the chunk text. v1 callers keep using
#: ``SCHEMA_VERSION``.
SCHEMA_VERSION_V2 = 2

#: Symbolic names matching the ontology's ``extraction_prompt`` values.
ENTITY_PROMPT_NAME = "extract_entities_v1"
RELATION_PROMPT_NAME = "extract_relations_v1"


def load_ontology(path: str | Path) -> dict:
    """Load ``ontology.yaml`` as a plain mapping.

    The generic ontology validator (``src.ontology.validate``) is the
    pre-flight shape gate; extraction only needs the loaded document.
    """
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def extracted_entity_types(ontology: dict) -> list[dict]:
    """Return ``entity_types`` filtered to ``population_mode == extracted``.

    Order is preserved from the YAML. Per ontology spec §4.3, the
    implementation never enumerates the v1 vocabulary names directly —
    whatever the ontology declares as ``extracted`` is the universe.
    """
    out: list[dict] = []
    for et in ontology.get("entity_types", []):
        if et.get("population_mode") == "extracted":
            out.append(et)
    return out


def extracted_relation_types(ontology: dict) -> list[dict]:
    """Return ``relation_types`` filtered to ``population_mode == extracted``.

    ``derived`` and ``structural`` relations are never emitted by the
    extraction prompt; ``EVALUATED_ON`` (derived) and ``CITES``
    (structural) are not in this list per spec §5.3.
    """
    out: list[dict] = []
    for rt in ontology.get("relation_types", []):
        if rt.get("population_mode") == "extracted":
            out.append(rt)
    return out


#: Allowed confidence levels for mentions and assertions.
CONFIDENCE_VALUES = ("stated", "inferred", "hedged")

#: Allowed endpoint kinds.
ENDPOINT_KINDS = ("document", "mention")


def entity_pass_schema(ontology: dict) -> dict:
    """JSON Schema for the entity-pass output shape.

    Top-level: ``{"mentions": [...]}`` with each mention shaped per
    spec §5.2. ``entity_type`` is constrained to the closed set of
    extracted ontology entity names; ``confidence`` is constrained to
    the three literal values per spec §3.1.1.
    """
    et_names = [et["name"] for et in extracted_entity_types(ontology)]
    mention = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "entity_type",
            "surface_form",
            "canonical_form",
            "char_start",
            "char_end",
            "confidence",
        ],
        "properties": {
            "entity_type": {"type": "string", "enum": et_names},
            "surface_form": {"type": "string", "minLength": 1},
            "canonical_form": {"type": "string", "minLength": 1},
            "char_start": {"type": "integer", "minimum": 0},
            "char_end": {"type": "integer", "minimum": 1},
            "confidence": {"type": "string", "enum": list(CONFIDENCE_VALUES)},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["mentions"],
        "properties": {"mentions": {"type": "array", "items": mention}},
    }


def relation_pass_schema(ontology: dict) -> dict:
    """JSON Schema for the relation-pass output shape.

    Top-level: ``{"assertions": [...]}`` with each assertion shaped
    per spec §5.3. ``relation_type`` is constrained to the closed set
    of extracted ontology relation names; endpoint ``kind`` is the
    two-literal set ``document|mention`` per spec §3.1.2.
    """
    rt_names = [rt["name"] for rt in extracted_relation_types(ontology)]
    endpoint = {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind", "id"],
        "properties": {
            "kind": {"type": "string", "enum": list(ENDPOINT_KINDS)},
            "id": {"type": "string", "minLength": 1},
        },
    }
    assertion = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "relation_type",
            "subject_ref",
            "object_ref",
            "char_start",
            "char_end",
            "confidence",
        ],
        "properties": {
            "relation_type": {"type": "string", "enum": rt_names},
            "subject_ref": endpoint,
            "object_ref": endpoint,
            "char_start": {"type": "integer", "minimum": 0},
            "char_end": {"type": "integer", "minimum": 1},
            "confidence": {"type": "string", "enum": list(CONFIDENCE_VALUES)},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["assertions"],
        "properties": {"assertions": {"type": "array", "items": assertion}},
    }


def entity_pass_v2_schema(ontology: dict) -> dict:
    """JSON Schema for the v2 entity-pass output shape (#51).

    Top-level: ``{"mentions": [...]}``. The model does not author
    ``char_start`` / ``char_end``; the deterministic exact-span v2
    aligner (#49) derives those offsets from the chunk text and an
    optional ``occurrence_index``. ``surface_form`` is the literal
    span the aligner locates; ``canonical_form`` is the LLM's
    normalised name; ``confidence`` is the closed set from spec
    §3.1.1. ``occurrence_index`` is an optional non-negative integer
    index into the ordered list of identical literals in the chunk
    text.
    """
    et_names = [et["name"] for et in extracted_entity_types(ontology)]
    mention = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "entity_type",
            "surface_form",
            "canonical_form",
            "confidence",
        ],
        "properties": {
            "entity_type": {"type": "string", "enum": et_names},
            "surface_form": {"type": "string", "minLength": 1},
            "canonical_form": {"type": "string", "minLength": 1},
            "confidence": {"type": "string", "enum": list(CONFIDENCE_VALUES)},
            "occurrence_index": {"type": "integer", "minimum": 0},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["mentions"],
        "properties": {"mentions": {"type": "array", "items": mention}},
    }


def relation_pass_v2_schema(ontology: dict) -> dict:
    """JSON Schema for the v2 relation-pass output shape (#51).

    Top-level: ``{"assertions": [...]}``. The model does not author
    ``char_start`` / ``char_end``; the deterministic exact-span v2
    aligner (#49) derives those offsets from the chunk text and an
    optional ``occurrence_index``. ``evidence_quote`` is required
    and is the literal span the aligner locates; endpoint ``kind``
    is the two-literal set ``document|mention`` per spec §3.1.2.
    """
    rt_names = [rt["name"] for rt in extracted_relation_types(ontology)]
    endpoint = {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind", "id"],
        "properties": {
            "kind": {"type": "string", "enum": list(ENDPOINT_KINDS)},
            "id": {"type": "string", "minLength": 1},
        },
    }
    assertion = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "relation_type",
            "subject_ref",
            "object_ref",
            "evidence_quote",
            "confidence",
        ],
        "properties": {
            "relation_type": {"type": "string", "enum": rt_names},
            "subject_ref": endpoint,
            "object_ref": endpoint,
            "evidence_quote": {"type": "string", "minLength": 1},
            "confidence": {"type": "string", "enum": list(CONFIDENCE_VALUES)},
            "occurrence_index": {"type": "integer", "minimum": 0},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["assertions"],
        "properties": {"assertions": {"type": "array", "items": assertion}},
    }


def entity_type_endpoint_table(ontology: dict) -> dict[str, str]:
    """Return a mapping ``entity_type_name -> population_mode``.

    Used by the relation endpoint validation gate (spec §8.2) to
    decide whether an endpoint ``kind`` matches the relation's from/to
    endpoint type.
    """
    return {
        et["name"]: et.get("population_mode")
        for et in ontology.get("entity_types", [])
    }


def relation_endpoints(ontology: dict) -> dict[str, tuple[str, str]]:
    """Return ``relation_type_name -> (from_entity_type, to_entity_type)``.

    Used by the relation-pass validation gate (spec §8.2): each
    assertion's ``subject_ref`` must satisfy the from slot's endpoint
    type, and ``object_ref`` must satisfy the to slot's endpoint type.
    """
    return {
        rt["name"]: (rt["from"], rt["to"])
        for rt in ontology.get("relation_types", [])
    }


def build_allowed_name_lists(ontology: dict) -> dict[str, list[str]]:
    """Build the per-template allowed-name lists rendered into prompts.

    Returns ``{"entity_types": [...], "relation_types": [...]}``. The
    lists are the closed sets used in the rendered prompt text per
    spec §5.2 / §5.3 — never hard-coded in Python.
    """
    return {
        "entity_types": [et["name"] for et in extracted_entity_types(ontology)],
        "relation_types": [rt["name"] for rt in extracted_relation_types(ontology)],
    }


def verified_examples_block(ontology: dict) -> str:
    """Render the per-type ``verified`` examples for the entity prompt.

    Per spec §5.2, examples in the prompt come from
    ``ontology.yaml``'s ``examples[].status == "verified"``. The output
    is plain prose (joined bullets); the prompt text embeds it
    verbatim, with the understanding that the ontology already
    validates verified examples against the canonical manifest.
    """
    out: list[str] = []
    for et in extracted_entity_types(ontology):
        name = et.get("name", "<unnamed>")
        definition = et.get("definition", "")
        out.append(f"- {name}: {definition}")
        for ex in et.get("examples", []):
            if ex.get("status") == "verified":
                instance = ex.get("instance", "<unspecified>")
                note = ex.get("note", "")
                if note:
                    out.append(f"    - verified example: {instance} ({note})")
                else:
                    out.append(f"    - verified example: {instance}")
    return "\n".join(out) if out else "(no verified examples available)"


def relation_definitions_block(ontology: dict) -> str:
    """Render relation endpoint / definition prose for the relation prompt."""
    out: list[str] = []
    for rt in extracted_relation_types(ontology):
        name = rt.get("name", "<unnamed>")
        from_t = rt.get("from", "?")
        to_t = rt.get("to", "?")
        definition = rt.get("definition", "")
        out.append(
            f"- {name} ({from_t} -> {to_t}): {definition}"
        )
    return "\n".join(out) if out else "(no extracted relations defined)"