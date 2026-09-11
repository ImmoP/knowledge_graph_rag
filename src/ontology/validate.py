"""Generic YAML-driven validator for ``ontology.yaml`` (#20).

Per ``docs/specs/ontology.md`` §4.3 the v1 entity / relation vocabulary,
endpoints, and population modes live **only** in ``ontology.yaml``. This
validator therefore never enumerates that vocabulary in Python; it loads
the YAML and checks the **shape** of the file against the spec, plus the
two validation-policy surfaces that are intentionally not vocabulary:

* the three literal ``population_mode`` names;
* the two phase-specific symbolic prompt names
  (``extract_entities_v1`` for extracted entity types,
  ``extract_relations_v1`` for extracted relation types, per
  ``docs/specs/extraction-contract.md`` §5);
* the reserved / forbidden entity-type names from ontology.md §1.3.

Verified-example corroboration uses the canonical pre-chunk artifacts:
the corpus manifest (``data/manifest.jsonl``) and the raw OpenAlex
metadata under ``data/raw/openalex/``. ``data/derived/chunks.jsonl``
is **not** consulted; the validator must run before the chunker (#22).

Public entry points:

* :func:`validate_ontology` — collect-all validation returning a
  :class:`ValidationReport`.
* :func:`main` — CLI wrapper exiting non-zero on any violation.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
ONTOLOGY_PATH = ROOT / "ontology.yaml"
MANIFEST_PATH = ROOT / "data" / "manifest.jsonl"
OPENALEX_DIR = ROOT / "data" / "raw" / "openalex"


# --- Validation-policy constants (NOT v1 vocabulary) ----------------

POPULATION_MODES = ("structural", "extracted", "derived")

ENTITY_EXTRACTION_PROMPT = "extract_entities_v1"
RELATION_EXTRACTION_PROMPT = "extract_relations_v1"
EXTRACTION_PROMPTS = (ENTITY_EXTRACTION_PROMPT, RELATION_EXTRACTION_PROMPT)

FORBIDDEN_ENTITY_NAMES = (
    "author",
    "concept",
    "finding",
    "claim",
    "benchmark",
    "topic",
    "theme",
)

ENTITY_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
RELATION_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
OPENALEX_ID_RE = re.compile(r"^W[0-9]+$")


def _looks_like_openalex_id(s: str) -> bool:
    """Return ``True`` iff ``s`` matches the OpenAlex work-id convention.

    Used as the generic structural cue for the manifest-membership check
    on verified entity-type examples (per ontology.md §5.9 — the document
    instance is the manifest ``openalex_id``). No v1 vocabulary required.
    """
    return isinstance(s, str) and bool(OPENALEX_ID_RE.match(s))

ALLOWED_TOP_LEVEL_KEYS = frozenset(
    {"version", "schema_version", "entity_types", "relation_types", "notes"}
)
ALLOWED_EXAMPLE_STATUSES = frozenset({"verified", "pending_verification"})

ENTITY_MIN, ENTITY_MAX = 5, 10
RELATION_MIN, RELATION_MAX = 8, 15


# --- Report types ----------------------------------------------------


@dataclass(frozen=True)
class Violation:
    rule: str
    record: str
    message: str

    def format(self) -> str:
        return f"[{self.rule}] {self.record}: {self.message}"


@dataclass(frozen=True)
class ValidationReport:
    yaml_path: Path
    yaml_sha256: str
    version: int | None = None
    schema_version: int | None = None
    violations: list[Violation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


# --- Corpus surface --------------------------------------------------


@dataclass(frozen=True)
class CorpusSurface:
    """Pre-chunk corpus surface for verified-example corroboration.

    ``manifest_ids`` powers the document membership check (§5.9);
    ``surface`` is the corpus-wide title + reconstructed abstract text
    lower-cased for substring matching (§5.9 + extraction-contract.md §5).
    """

    manifest_ids: frozenset = field(default_factory=frozenset)
    surface: str = ""
    surfaces_by_id: dict = field(default_factory=dict)

    def contains(self, needle: str) -> bool:
        return needle.lower() in self.surface

    def manifest_has(self, openalex_id: str) -> bool:
        return openalex_id in self.manifest_ids


def _load_manifest_ids(path: Path) -> frozenset:
    if not path.exists():
        return frozenset()
    ids: set = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            oid = rec.get("openalex_id")
            if isinstance(oid, str):
                ids.add(oid)
    return frozenset(ids)


def _reconstruct_abstract(aii) -> str:
    """Reconstruct OpenAlex ``abstract_inverted_index`` → plain text."""
    if not aii:
        return ""
    positions: dict = {}
    for token, idxs in aii.items():
        if not isinstance(idxs, list):
            continue
        for i in idxs:
            if isinstance(i, int):
                positions[i] = token
    if not positions:
        return ""
    return " ".join(positions[i] for i in range(max(positions) + 1))


def _build_surface(openalex_dir: Path):
    chunks: list = []
    per_id: dict = {}
    if not openalex_dir.is_dir():
        return "", per_id
    for fp in sorted(openalex_dir.glob("*.json")):
        oid = fp.stem
        try:
            meta = json.loads(fp.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(meta, dict):
            continue
        bits: list = []
        title = meta.get("title")
        if isinstance(title, str) and title:
            bits.append(title)
        display = meta.get("display_name")
        if isinstance(display, str) and display and display != title:
            bits.append(display)
        abstract = _reconstruct_abstract(meta.get("abstract_inverted_index"))
        if abstract:
            bits.append(abstract)
        if bits:
            paper = " ".join(bits)
            per_id[oid] = paper.lower()
            chunks.append(paper)
    return " ".join(chunks).lower(), per_id


def build_corpus_surface(manifest_path=None, openalex_dir=None) -> CorpusSurface:
    """Load the corpus surface from the canonical pre-chunk artifacts."""
    mp = Path(manifest_path) if manifest_path is not None else MANIFEST_PATH
    od = Path(openalex_dir) if openalex_dir is not None else OPENALEX_DIR
    blob, per_id = _build_surface(od)
    return CorpusSurface(
        manifest_ids=_load_manifest_ids(mp),
        surface=blob,
        surfaces_by_id=per_id,
    )


# --- Schema validation ----------------------------------------------


def _is_nonempty_str(x) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _check_top_level(doc) -> tuple:
    warnings: list = []
    violations: list = []
    if not isinstance(doc, dict):
        violations.append(
            Violation(
                "top_level_mapping",
                "$",
                "top-level YAML document must be a mapping",
            )
        )
        return {}, violations, warnings
    extra = set(doc.keys()) - ALLOWED_TOP_LEVEL_KEYS
    for k in sorted(extra):
        violations.append(
            Violation(
                "top_level_keys",
                f"$.{k}",
                f"unexpected top-level key '{k}'",
            )
        )
    version = doc.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        violations.append(
            Violation(
                "version",
                "$.version",
                f"version must be a positive int; got {version!r}",
            )
        )
    schema_version = doc.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version < 1
    ):
        violations.append(
            Violation(
                "schema_version",
                "$.schema_version",
                f"schema_version must be a positive int; got {schema_version!r}",
            )
        )
    if doc.get("notes") is not None and not _is_nonempty_str(doc["notes"]):
        violations.append(
            Violation(
                "notes",
                "$.notes",
                "notes must be a non-empty string when present",
            )
        )
    return doc, violations, warnings


_MODE_FIELDS = {
    "structural": (frozenset({"source"}), "source"),
    "extracted": (frozenset({"extraction_prompt"}), "extraction_prompt"),
    "derived": (frozenset({"derivation"}), "derivation"),
}


def _check_mode_fields(rec: dict, kind: str, name: str, record_path: str, *, expected_prompt):
    out: list = []
    mode = rec.get("population_mode")
    allowed, _required = _MODE_FIELDS.get(mode, (frozenset(), ""))
    if not allowed:
        return out
    for k in ("source", "extraction_prompt", "derivation"):
        if k in rec and k not in allowed:
            out.append(
                Violation(
                    "5.5",
                    f"{record_path}.{k}",
                    f"field '{k}' is not allowed for population_mode={mode!r} "
                    f"({kind} {name!r})",
                )
            )
    for k in allowed:
        if k not in rec:
            out.append(
                Violation(
                    "5.5",
                    f"{record_path}.{k}",
                    f"field '{k}' is required for population_mode={mode!r} "
                    f"({kind} {name!r})",
                )
            )
    if mode == "extracted" and expected_prompt is not None:
        actual = rec.get("extraction_prompt")
        if actual != expected_prompt:
            if actual in EXTRACTION_PROMPTS:
                out.append(
                    Violation(
                        "5.10",
                        f"{record_path}.extraction_prompt",
                        f"phase-specific prompt swap: expected {expected_prompt!r} "
                        f"for {kind} {name!r}, got {actual!r}",
                    )
                )
            else:
                out.append(
                    Violation(
                        "5.10",
                        f"{record_path}.extraction_prompt",
                        f"unknown extraction_prompt {actual!r} for {kind} {name!r}; "
                        f"must equal {expected_prompt!r}",
                    )
                )
    return out


def _check_example_shape(ex, path: str, expected_keys) -> list:
    out: list = []
    if not isinstance(ex, dict):
        out.append(Violation("example_shape", path, "example must be a mapping"))
        return out
    extras = set(ex.keys()) - expected_keys
    if extras:
        out.append(
            Violation(
                "example_shape",
                path,
                f"unexpected example fields: {sorted(extras)}",
            )
        )
    missing = expected_keys - set(ex.keys())
    if missing:
        out.append(
            Violation(
                "example_shape",
                path,
                f"missing example fields: {sorted(missing)}",
            )
        )
    return out


# --- Record validators ----------------------------------------------


def _check_entity_record(rec, seen_names: set, surface: CorpusSurface):
    out: list = []
    if not isinstance(rec, dict):
        return None, [Violation("record_shape", "$", "entity record must be a mapping")]
    raw_name = rec.get("name")
    path = f"$.entity_types[{raw_name if isinstance(raw_name, str) else '?'}]"
    name_for_path = raw_name if isinstance(raw_name, str) else "?"
    name = None
    if not _is_nonempty_str(raw_name) or not ENTITY_NAME_RE.match(raw_name or ""):
        out.append(
            Violation(
                "entity_name",
                f"{path}.name",
                f"entity name must match ^[a-z][a-z0-9_]*$; got {raw_name!r}",
            )
        )
    else:
        name = raw_name
        if name in seen_names:
            out.append(
                Violation(
                    "entity_unique",
                    f"{path}.name",
                    f"duplicate entity name {name!r}",
                )
            )
        seen_names.add(name)
    if isinstance(raw_name, str) and raw_name.lower() in FORBIDDEN_ENTITY_NAMES:
        out.append(
            Violation(
                "1.3",
                f"{path}.name",
                f"entity type {raw_name!r} is forbidden (name-discipline)",
            )
        )
    definition = rec.get("definition")
    if not _is_nonempty_str(definition):
        out.append(
            Violation(
                "definition",
                f"{path}.definition",
                "definition must be a non-empty string",
            )
        )
    mode = rec.get("population_mode")
    if mode not in POPULATION_MODES:
        out.append(
            Violation(
                "population_mode",
                f"{path}.population_mode",
                f"population_mode must be one of {list(POPULATION_MODES)}; got {mode!r}",
            )
        )
    out.extend(
        _check_mode_fields(
            rec,
            "entity type",
            name_for_path,
            path,
            expected_prompt=ENTITY_EXTRACTION_PROMPT if mode == "extracted" else None,
        )
    )
    examples = rec.get("examples")
    if not isinstance(examples, list) or not examples:
        out.append(
            Violation(
                "examples",
                f"{path}.examples",
                "examples must be a non-empty list",
            )
        )
    else:
        for idx, ex in enumerate(examples):
            out.extend(
                _check_example_shape(
                    ex,
                    f"{path}.examples[{idx}]",
                    {"status", "instance", "note"},
                )
            )
            if not isinstance(ex, dict):
                continue
            status = ex.get("status")
            instance = ex.get("instance")
            note = ex.get("note")
            if status not in ALLOWED_EXAMPLE_STATUSES:
                out.append(
                    Violation(
                        "example_status",
                        f"{path}.examples[{idx}].status",
                        f"status must be one of {sorted(ALLOWED_EXAMPLE_STATUSES)}; "
                        f"got {status!r}",
                    )
                )
                continue
            if not _is_nonempty_str(instance):
                out.append(
                    Violation(
                        "example_instance",
                        f"{path}.examples[{idx}].instance",
                        "instance must be a non-empty string",
                    )
                )
            if status == "pending_verification":
                if not _is_nonempty_str(note):
                    out.append(
                        Violation(
                            "5.10",
                            f"{path}.examples[{idx}].note",
                            "pending_verification example MUST carry a non-empty note",
                        )
                    )
            elif status == "verified" and name is not None and isinstance(instance, str):
                # Generic structural check: an instance string that matches
                # the OpenAlex work-id convention ``^W[0-9]+$`` is treated
                # as a manifest-anchored identifier and must appear in the
                # corpus manifest. Everything else is a corpus-surface
                # claim that must appear in at least one paper's
                # reconstructed title/abstract surface.
                if _looks_like_openalex_id(instance):
                    if not surface.manifest_has(instance):
                        out.append(
                            Violation(
                                "5.9",
                                f"{path}.examples[{idx}].instance",
                                f"verified manifest-anchored example instance "
                                f"{instance!r} not in manifest",
                            )
                        )
                else:
                    if not surface.contains(instance):
                        out.append(
                            Violation(
                                "5.9",
                                f"{path}.examples[{idx}].instance",
                                f"verified non-document example instance {instance!r} "
                                "absent from reconstructed title/abstract surface",
                            )
                        )
    return name, out


def _check_relation_record(
    rec, seen_names: set, entity_names: set, surface: CorpusSurface
):
    out: list = []
    if not isinstance(rec, dict):
        return None, [
            Violation("record_shape", "$", "relation record must be a mapping")
        ]
    raw_name = rec.get("name")
    path = f"$.relation_types[{raw_name if isinstance(raw_name, str) else '?'}]"
    name_for_path = raw_name if isinstance(raw_name, str) else "?"
    name = None
    if not _is_nonempty_str(raw_name) or not RELATION_NAME_RE.match(raw_name or ""):
        out.append(
            Violation(
                "relation_name",
                f"{path}.name",
                f"relation name must match ^[A-Z][A-Z0-9_]*$; got {raw_name!r}",
            )
        )
    else:
        name = raw_name
        if name in seen_names:
            out.append(
                Violation(
                    "relation_unique",
                    f"{path}.name",
                    f"duplicate relation name {name!r}",
                )
            )
        seen_names.add(name)
    definition = rec.get("definition")
    if not _is_nonempty_str(definition):
        out.append(
            Violation(
                "definition",
                f"{path}.definition",
                "definition must be a non-empty string",
            )
        )
    mode = rec.get("population_mode")
    if mode not in POPULATION_MODES:
        out.append(
            Violation(
                "population_mode",
                f"{path}.population_mode",
                f"population_mode must be one of {list(POPULATION_MODES)}; got {mode!r}",
            )
        )
    from_ = rec.get("from")
    if not _is_nonempty_str(from_):
        out.append(
            Violation(
                "5.4",
                f"{path}.from",
                "from must be a non-empty string naming an entity type",
            )
        )
    elif from_ not in entity_names:
        out.append(
            Violation(
                "5.4",
                f"{path}.from",
                f"from {from_!r} does not name an entity type",
            )
        )
    to_ = rec.get("to")
    if not _is_nonempty_str(to_):
        out.append(
            Violation(
                "5.4",
                f"{path}.to",
                "to must be a non-empty string naming an entity type",
            )
        )
    elif to_ not in entity_names:
        out.append(
            Violation(
                "5.4",
                f"{path}.to",
                f"to {to_!r} does not name an entity type",
            )
        )
    out.extend(
        _check_mode_fields(
            rec,
            "relation type",
            name_for_path,
            path,
            expected_prompt=RELATION_EXTRACTION_PROMPT if mode == "extracted" else None,
        )
    )
    examples = rec.get("examples")
    if not isinstance(examples, list) or not examples:
        out.append(
            Violation(
                "examples",
                f"{path}.examples",
                "examples must be a non-empty list",
            )
        )
    else:
        for idx, ex in enumerate(examples):
            out.extend(
                _check_example_shape(
                    ex,
                    f"{path}.examples[{idx}]",
                    {"status", "from", "to", "note"},
                )
            )
            if not isinstance(ex, dict):
                continue
            status = ex.get("status")
            if status not in ALLOWED_EXAMPLE_STATUSES:
                out.append(
                    Violation(
                        "example_status",
                        f"{path}.examples[{idx}].status",
                        f"status must be one of {sorted(ALLOWED_EXAMPLE_STATUSES)}; "
                        f"got {status!r}",
                    )
                )
                continue
            from_ex = ex.get("from")
            to_ex = ex.get("to")
            note = ex.get("note")
            if not _is_nonempty_str(from_ex):
                out.append(
                    Violation(
                        "example_from",
                        f"{path}.examples[{idx}].from",
                        "from must be a non-empty string",
                    )
                )
            if not _is_nonempty_str(to_ex):
                out.append(
                    Violation(
                        "example_to",
                        f"{path}.examples[{idx}].to",
                        "to must be a non-empty string",
                    )
                )
            if status == "pending_verification":
                if not _is_nonempty_str(note):
                    out.append(
                        Violation(
                            "5.10",
                            f"{path}.examples[{idx}].note",
                            "pending_verification example MUST carry a non-empty note",
                        )
                    )
                continue
            # verified: per §5.9 the example `to` is an entity-instance
            # identifier and must surface corpus-wide; the `from` end is
            # checked against the manifest iff it looks like a document
            # openalex_id (starts with 'W'). Non-document `from` references
            # (method, dataset, metric, task) are entity-name anchors that
            # the extraction ticket will later corroborate against full
            # chunk text — verified status here only vouches for the
            # relation example as a whole, not for the `from` anchor's
            # separate existence (which is an entity-type concern).
            if isinstance(from_ex, str) and _looks_like_openalex_id(from_ex):
                if not surface.manifest_has(from_ex):
                    out.append(
                        Violation(
                            "5.9",
                            f"{path}.examples[{idx}].from",
                            f"verified relation from {from_ex!r} not in manifest",
                        )
                    )
            if isinstance(to_ex, str) and not surface.contains(to_ex):
                out.append(
                    Violation(
                        "5.9",
                        f"{path}.examples[{idx}].to",
                        f"verified relation to {to_ex!r} absent from reconstructed "
                        "title/abstract surface",
                    )
                )
    return name, out


# --- Public entry points --------------------------------------------


def validate_ontology(
    yaml_path=None,
    *,
    manifest_path=None,
    openalex_dir=None,
    on_vocab_bump="warn",
) -> ValidationReport:
    """Validate ``yaml_path`` (default: repo-root ``ontology.yaml``).

    Collect-all semantics: every rule violation is reported, even after
    the first failure. The CLI exits non-zero iff ``report.ok`` is False.

    ``manifest_path`` and ``openalex_dir`` are injectable so tests can
    drive the corpus surface from synthetic fixtures without depending
    on the canonical pre-chunk artifacts.

    ``on_vocab_bump`` controls the version-bump guard from #20:
    ``"warn"`` (default) records a warning when ``version > 1``;
    ``"ignore"`` skips the warning; ``"error"`` records a violation.
    """
    path = Path(yaml_path) if yaml_path is not None else ONTOLOGY_PATH
    raw = path.read_text()
    sha = hashlib.sha256(raw.encode()).hexdigest()
    try:
        doc = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        return ValidationReport(
            yaml_path=path,
            yaml_sha256=sha,
            violations=[
                Violation(
                    "yaml_parse",
                    "$",
                    f"YAML parse error: {e}",
                )
            ],
        )
    doc, tl_violations, warnings = _check_top_level(doc)
    version = doc.get("version")
    schema_version = doc.get("schema_version")
    norm_version = (
        version if isinstance(version, int) and not isinstance(version, bool) else None
    )
    norm_schema = (
        schema_version
        if isinstance(schema_version, int) and not isinstance(schema_version, bool)
        else None
    )
    report = ValidationReport(
        yaml_path=path,
        yaml_sha256=sha,
        version=norm_version,
        schema_version=norm_schema,
        violations=list(tl_violations),
        warnings=list(warnings),
    )
    # Version-bump guard (#20, behaviour differs by on_vocab_bump).
    if report.version is not None and report.version > 1:
        msg = (
            f"ontology version {report.version} > 1 — vocabulary bump detected; "
            "confirm downstream re-run before accepting (§1.4)"
        )
        if on_vocab_bump == "error":
            report.violations.append(
                Violation("1.4", "$.version", msg)
            )
        elif on_vocab_bump == "warn":
            report.warnings.append(msg)
    if report.schema_version is not None and report.schema_version > 1:
        msg = (
            f"schema_version {report.schema_version} > 1 — schema bump detected; "
            "validator may need update before accepting"
        )
        if on_vocab_bump == "error":
            report.violations.append(
                Violation("4.1", "$.schema_version", msg)
            )
        elif on_vocab_bump == "warn":
            report.warnings.append(msg)

    surface = build_corpus_surface(
        manifest_path=manifest_path, openalex_dir=openalex_dir
    )

    entity_records = doc.get("entity_types")
    seen_entity_names: set = set()
    entity_names: set = set()
    if not isinstance(entity_records, list) or not entity_records:
        report.violations.append(
            Violation(
                "entity_types",
                "$.entity_types",
                "entity_types must be a non-empty list",
            )
        )
    else:
        n = len(entity_records)
        if n < ENTITY_MIN or n > ENTITY_MAX:
            report.violations.append(
                Violation(
                    "1.1",
                    "$.entity_types",
                    f"entity_types must have {ENTITY_MIN}..{ENTITY_MAX} entries; got {n}",
                )
            )
        for rec in entity_records:
            name, vs = _check_entity_record(rec, seen_entity_names, surface)
            report.violations.extend(vs)
            if name:
                entity_names.add(name)

    relation_records = doc.get("relation_types")
    seen_relation_names: set = set()
    if not isinstance(relation_records, list) or not relation_records:
        report.violations.append(
            Violation(
                "relation_types",
                "$.relation_types",
                "relation_types must be a non-empty list",
            )
        )
    else:
        n = len(relation_records)
        if n < RELATION_MIN or n > RELATION_MAX:
            report.violations.append(
                Violation(
                    "1.2",
                    "$.relation_types",
                    f"relation_types must have {RELATION_MIN}..{RELATION_MAX} entries; "
                    f"got {n}",
                )
            )
        for rec in relation_records:
            _, vs = _check_relation_record(
                rec, seen_relation_names, entity_names, surface
            )
            report.violations.extend(vs)

    # If entity-type validation has critical failures we may have an
    # inconsistent `entity_names` set; relation checks already guarded
    # their from/to lookups against `entity_names` directly.

    return report


def _format_report(report: ValidationReport) -> str:
    lines: list = []
    lines.append(f"ontology: {report.yaml_path}")
    lines.append(f"sha256:   {report.yaml_sha256}")
    lines.append(
        f"version:  {report.version}  schema_version: {report.schema_version}"
    )
    n_v = len(report.violations)
    n_w = len(report.warnings)
    lines.append(f"violations: {n_v}  warnings: {n_w}")
    if report.warnings:
        for w in report.warnings:
            lines.append(f"  WARN  {w}")
    if report.violations:
        for v in report.violations:
            lines.append(f"  FAIL  {v.format()}")
    return "\n".join(lines)


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--help" in args or "-h" in args:
        print(
            "usage: python -m src.ontology.validate [path-to-ontology.yaml]\n"
            "\n"
            "Validates ontology.yaml against docs/specs/ontology.md §§4–5.\n"
            "Exits non-zero with the first violation surfaced per rule.\n"
            "Optional env overrides (used by tests):\n"
            "  ONTOLOGY_MANIFEST_PATH  path to a manifest.jsonl\n"
            "  ONTOLOGY_OPENALEX_DIR   path to a directory of OpenAlex metadata .json\n"
        )
        return 0
    yaml_path = Path(args[0]) if args else None
    manifest_path = (
        Path(p) if (p := __import__("os").environ.get("ONTOLOGY_MANIFEST_PATH")) else None
    )
    openalex_dir = (
        Path(p) if (p := __import__("os").environ.get("ONTOLOGY_OPENALEX_DIR")) else None
    )
    report = validate_ontology(
        yaml_path,
        manifest_path=manifest_path,
        openalex_dir=openalex_dir,
    )
    print(_format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
