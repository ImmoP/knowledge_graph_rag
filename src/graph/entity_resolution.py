"""Deterministic two-stage entity resolution (#24, #8).

Implements ``docs/specs/entity-resolution.md`` and impl-spec §8
verbatim. The resolver is pure-Python and deterministic. It does
not call LLMProvider, EmbeddingProvider, or any provider SDK; it
does not build an embedding or similarity index.

Pipeline
--------

1. **Preflight** (§2.1): the extraction manifest must exist and
   agree on SHA256 with the on-disk ``chunks.jsonl``,
   ``extractions.jsonl``, and ``ontology.yaml``; every chunk in
   ``chunks.jsonl`` must have exactly one ``chunk_status`` record
   with ``extraction_complete: true``; no extraction record may
   reference an unknown or stale ``chunk_id``.

2. **Eligible mention set** (§2.2): schema-valid ``mention``
   records whose chunk is extraction-complete, whose
   ``entity_type`` is an ontology ``extracted`` type, and whose
   ``mention_id`` re-derives from the record's evidence fields.

3. **Stage A** (§5.1): group eligible mentions by
   ``(entity_type, normalized_resolution_name)``.

4. **Stage B** (§5.2): deterministic greedy complete-link within
   each entity type; threshold = 0.92 on normalised Levenshtein
   similarity.

5. **Outputs** (§7): ``data/entities.jsonl``,
   ``data/entity_resolutions.jsonl``,
   ``data/entity_resolution.manifest.json``.

The threshold, algorithm version, and normalisation version live
as named module-scope constants so a benchmark-producing run
cannot be reconfigured through CLI flags or environment
variables.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Named constants — the only knobs the resolver exposes. The threshold,
# algorithm version, and normalisation version are pinned in code; no CLI or
# environment override is honoured. See entity-resolution spec §4 and §4.1.
# ---------------------------------------------------------------------------

ENTITY_RESOLUTION_ALGORITHM_VERSION = "entity_resolution_v1"
NORMALIZATION_VERSION = "resolution_name_v1"
SIMILARITY_METRIC = "normalized_levenshtein"
NORMALIZED_LEVENSHTEIN_THRESHOLD = 0.92


# ---------------------------------------------------------------------------
# Path defaults — kept module-scope so callers and tests share them.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_PATH = REPO_ROOT / "data" / "derived" / "chunks.jsonl"
DEFAULT_EXTRACTIONS_PATH = REPO_ROOT / "data" / "extractions.jsonl"
DEFAULT_EXTRACTIONS_MANIFEST_PATH = REPO_ROOT / "data" / "extractions.manifest.json"
DEFAULT_ONTOLOGY_PATH = REPO_ROOT / "ontology.yaml"
DEFAULT_ENTITIES_PATH = REPO_ROOT / "data" / "entities.jsonl"
DEFAULT_ENTITY_RESOLUTIONS_PATH = REPO_ROOT / "data" / "entity_resolutions.jsonl"
DEFAULT_ENTITY_RESOLUTION_MANIFEST_PATH = REPO_ROOT / "data" / "entity_resolution.manifest.json"


# ---------------------------------------------------------------------------
# Pipeline-level result types.
# ---------------------------------------------------------------------------


class PreflightError(RuntimeError):
    """Raised when the §2.1 hard preflight fails. The resolver aborts
    before any output is written and leaves any prior valid
    generation untouched.
    """


@dataclass(frozen=True)
class ResolutionResult:
    """End-of-run summary returned by :func:`run_resolution`."""

    entities_records: int
    entity_resolutions_records: int
    eligible_mentions: int
    identical_duplicate_mentions: int
    extraction_complete_chunks: int


# ---------------------------------------------------------------------------
# Normalisation — §3 verbatim chain.
# ---------------------------------------------------------------------------

# Whitelist of paired wrapper-quote characters accepted by §3 step 6.
# Each entry maps an opener to its matching closer. ASCII / grave
# quotes are symmetric (the same character opens and closes), while
# curly quotes and guillemets come in matched open/close pairs. A
# wrapper pair is stripped ONLY when the first and last characters
# match one of these canonical (opener, closer) tuples — reversed
# pairs (e.g. ``”DPR“``) are NOT accepted and are preserved verbatim.
_WRAPPER_OPEN_TO_CLOSE: dict[str, str] = {
    "\"": "\"",        # ASCII double quote
    "'": "'",          # ASCII single quote
    "`": "`",          # grave accent
    "“": "”",          # left curly double quote  -> right curly double quote
    "‘": "’",          # left curly single quote  -> right curly single quote
    "«": "»",          # left guillemet            -> right guillemet
    "‹": "›",          # single left guillemet     -> single right guillemet
}

_WRAPPER_OPENERS: frozenset[str] = frozenset(_WRAPPER_OPEN_TO_CLOSE.keys())

_WHITESPACE_RE = re.compile(r"\s+")


def _map_pd_to_ascii_dash(s: str) -> str:
    """Map every Unicode Dash_Punctuation (Pd) character to ASCII ``-``."""
    return "".join("-" if unicodedata.category(ch) == "Pd" else ch for ch in s)


def _collapse_whitespace_and_trim(s: str) -> str:
    """Collapse every maximal whitespace run to one ASCII space and trim."""
    return _WHITESPACE_RE.sub(" ", s).strip()


def _strip_one_wrapper_pair(s: str) -> str:
    """If ``s`` begins and ends with a matched whitelist pair enclosing a
    non-empty value, return the inner value, re-trimmed; otherwise return
    ``s`` unchanged. See §3 step 6.
    """
    if len(s) < 2:
        return s
    first, last = s[0], s[-1]
    if first not in _WRAPPER_OPENERS:
        return s
    if _WRAPPER_OPEN_TO_CLOSE[first] != last:
        return s
    inner = s[1:-1]
    if not inner:
        return s
    return inner.strip()


def normalized_resolution_name(s: str) -> str:
    """Apply the §3 normalisation chain in its exact order.

    The chain is intentionally narrow and verbatim; it never stems,
    lemmatises, expands acronyms, transliterates, strips accents,
    removes stop words, reorders tokens, or strips punctuation other
    than the explicitly whitelisted wrapper-quote pairs.
    """
    if not isinstance(s, str):
        raise TypeError("normalized_resolution_name requires a str")
    if s == "":
        return ""
    out = unicodedata.normalize("NFKC", s)
    out = out.casefold()
    out = _map_pd_to_ascii_dash(out)
    out = _collapse_whitespace_and_trim(out)
    # Step 6 is repeated: strip a matched pair, re-trim, repeat until
    # neither end is a whitelist opener. The loop is bounded by len(s)/2
    # but in practice converges in 1–3 iterations.
    while True:
        stripped = _strip_one_wrapper_pair(out)
        if stripped == out:
            return out
        out = stripped


def display_normalize(s: str) -> str:
    """§6.2 display normalisation for canonical names and aliases.

    Only whitespace is touched: runs collapse to one ASCII space and
    edges are trimmed. Case, punctuation, digits, and Unicode
    characters are otherwise preserved.
    """
    return _WHITESPACE_RE.sub(" ", s).strip()

def lev_distance(a: str, b: str) -> int:
    """Plain Levenshtein edit distance over Unicode code points.

    Insertions, deletions, and substitutions each cost 1; transpositions
    are not a primitive. Lengths and edit operations are measured over
    code points (Python ``str`` indexing is code-point based).
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Standard two-row DP — O(len(a) * len(b)) time, O(min(len(a), len(b)))
    # memory. We swap so the inner loop walks the shorter string.
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def lev_similarity(a: str, b: str) -> float:
    """§4 normalised Levenshtein similarity in ``[0, 1]``."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    return 1.0 - lev_distance(a, b) / max(la, lb)


# ---------------------------------------------------------------------------
# §2.1 / §2.2 preflight + eligibility.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ChunkRecord:
    """Minimal chunk view used by preflight — pulls only the fields we need."""

    chunk_id: str
    doc_id: str


def _read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            yield json.loads(line)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_ontology(ontology_path: Path) -> dict:
    return yaml.safe_load(ontology_path.read_text(encoding="utf-8"))


def _extracted_entity_types(ontology: dict) -> set[str]:
    return {
        et["name"]
        for et in ontology.get("entity_types", [])
        if isinstance(et, dict) and et.get("population_mode") == "extracted"
    }


# Required fields on the extraction manifest (§2.1 / §13.1). The
# preflight enforces shape + types before any clustering runs.
_REQUIRED_MANIFEST_FIELDS: dict[str, type] = {
    "chunks_path": str,
    "chunks_sha256": str,
    "ontology_path": str,
    "ontology_sha256": str,
    "ontology_version": int,
    "schema_version": int,
    "provider": str,
    "model": str,
}


def _validate_manifest_shape(manifest: dict) -> None:
    """§2.1 manifest schema-validity gate.

    The manifest is a plain JSON object with a fixed field set;
    unknown types (e.g. ``None`` where a dict is required, list where
    a dict is required, missing required fields) abort resolution
    before any output is written.
    """
    if not isinstance(manifest, dict):
        raise PreflightError(
            f"extractions manifest must be a JSON object, got {type(manifest).__name__}"
        )
    missing = sorted(
        name for name, expected_type in _REQUIRED_MANIFEST_FIELDS.items()
        if name not in manifest
    )
    if missing:
        raise PreflightError(
            f"extractions manifest missing required fields: {missing}"
        )
    for name, expected_type in _REQUIRED_MANIFEST_FIELDS.items():
        if not isinstance(manifest[name], expected_type):
            raise PreflightError(
                f"extractions manifest field {name!r} must be {expected_type.__name__}, "
                f"got {type(manifest[name]).__name__}"
            )


def _preflight(
    *,
    chunks_path: Path,
    extractions_path: Path,
    extractions_manifest_path: Path,
    ontology_path: Path,
) -> dict:
    """§2.1 hard preflight. Returns a dict with the inputs needed by
    eligibility checks. Raises :class:`PreflightError` with a precise
    reason if any gate fails.
    """
    if not extractions_manifest_path.is_file():
        raise PreflightError(
            f"extractions manifest missing: {extractions_manifest_path}"
        )
    if not chunks_path.is_file():
        raise PreflightError(f"chunks file missing: {chunks_path}")
    if not extractions_path.is_file():
        raise PreflightError(f"extractions file missing: {extractions_path}")
    if not ontology_path.is_file():
        raise PreflightError(f"ontology file missing: {ontology_path}")

    try:
        manifest = json.loads(extractions_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PreflightError(
            f"extractions manifest is not valid JSON: {exc}"
        ) from exc
    _validate_manifest_shape(manifest)

    chunks_sha_on_disk = _file_sha256(chunks_path)
    if manifest["chunks_sha256"] != chunks_sha_on_disk:
        raise PreflightError(
            "extractions manifest chunks_sha256 disagrees with current "
            f"data/derived/chunks.jsonl (expected {chunks_sha_on_disk}, "
            f"got {manifest['chunks_sha256']})"
        )
    # §2.1 rule 3: the manifest's chunk-input path must point at the
    # same on-disk file the resolver is reading. Compare absolute
    # resolved paths so temporary/symlinked locations match.
    manifest_chunks_path = Path(manifest["chunks_path"]).resolve()
    if manifest_chunks_path != chunks_path.resolve():
        raise PreflightError(
            "extractions manifest chunks_path disagrees with current "
            f"chunks input (expected {str(chunks_path.resolve())!r}, "
            f"got {str(manifest_chunks_path)!r})"
        )

    ontology_sha_on_disk = _file_sha256(ontology_path)
    if manifest["ontology_sha256"] != ontology_sha_on_disk:
        raise PreflightError(
            "extractions manifest ontology_sha256 disagrees with current "
            f"ontology.yaml (expected {ontology_sha_on_disk}, "
            f"got {manifest['ontology_sha256']})"
        )

    ontology = _load_ontology(ontology_path)
    if not isinstance(ontology, dict):
        raise PreflightError(
            f"ontology must be a YAML mapping, got {type(ontology).__name__}"
        )
    ontology_version = ontology.get("version")
    if not isinstance(ontology_version, int):
        raise PreflightError(
            f"ontology 'version' must be an int, got {type(ontology_version).__name__}"
        )
    if manifest["ontology_version"] != ontology_version:
        raise PreflightError(
            "extractions manifest ontology_version disagrees with current "
            f"ontology.yaml (expected {ontology_version}, "
            f"got {manifest['ontology_version']})"
        )

    # Index chunks + extractions in one pass each.
    chunks: dict[str, _ChunkRecord] = {}
    for row in _read_jsonl(chunks_path):
        chunks[row["chunk_id"]] = _ChunkRecord(
            chunk_id=row["chunk_id"], doc_id=row["doc_id"]
        )

    mentions: list[dict] = []
    chunk_statuses: dict[str, dict] = {}
    duplicate_status_cids: list[str] = []
    for row in _read_jsonl(extractions_path):
        if not isinstance(row, dict):
            raise PreflightError(
                "extraction record must be a JSON object"
            )
        kind = row.get("kind")
        cid = row.get("chunk_id")
        if cid not in chunks:
            raise PreflightError(
                f"extraction record references unknown chunk_id: {cid!r}"
            )
        if kind == "mention":
            mentions.append(row)
        elif kind == "chunk_status":
            if cid in chunk_statuses:
                duplicate_status_cids.append(cid)
            else:
                chunk_statuses[cid] = row
        elif kind == "assertion":
            # Assertions do not participate in clustering, but their
            # chunk must still resolve to a real chunk. Already enforced
            # above.
            continue
        else:
            raise PreflightError(f"unknown extraction record kind: {kind!r}")

    if duplicate_status_cids:
        raise PreflightError(
            f"duplicate chunk_status records for {len(duplicate_status_cids)} "
            f"chunk_id(s); first: {duplicate_status_cids[0]!r}"
        )

    # Every required chunk_id must have exactly one complete status record.
    missing = sorted(set(chunks) - set(chunk_statuses))
    if missing:
        raise PreflightError(
            f"missing chunk_status records for {len(missing)} chunks "
            f"(first: {missing[0]!r})"
        )
    incomplete = sorted(
        cid for cid, st in chunk_statuses.items() if not st.get("extraction_complete")
    )
    if incomplete:
        raise PreflightError(
            f"{len(incomplete)} chunks have extraction_complete != true "
            f"(first: {incomplete[0]!r})"
        )

    return {
        "manifest": manifest,
        "chunks": chunks,
        "mentions": mentions,
        "chunk_statuses": chunk_statuses,
        "ontology_path": ontology_path,
        "ontology_sha": ontology_sha_on_disk,
        "ontology_version": ontology_version,
    }

def _default_rederive_mention_id(record: dict) -> str:
    """Default mention-ID re-derivation: ``m_`` + first 16 hex of
    ``sha256`` over the canonical-JSON evidence payload
    (``chunk_id``, ``source_text_sha256``, ``entity_type``,
    ``char_start``, ``char_end``).

    Inlined in this module so the resolver never imports any upstream
    provider SDK — required by entity-resolution spec §9 and
    implementation spec §10.
    """
    evidence = record["evidence"]
    payload = {
        "chunk_id": record["chunk_id"],
        "source_text_sha256": evidence["source_text_sha256"],
        "entity_type": record["entity_type"],
        "char_start": evidence["char_start"],
        "char_end": evidence["char_end"],
    }
    digest = sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return "m_" + digest[:16]

# ---------------------------------------------------------------------------
# §6.1 entity ID derivation.
# ---------------------------------------------------------------------------


def _canonical_entity_payload(*, canonical_name: str, entity_type: str) -> bytes:
    """§6.1 canonical payload for ``entity_id`` derivation.

    The payload contains exactly two fields; ``canonical_name`` is the
    selected display name passed through the §3 normaliser.
    Serialisation follows the extraction contract's canonical-JSON
    rules: ``sort_keys=True``, ``separators=(",", ":")``,
    ``ensure_ascii=False``, UTF-8.
    """
    payload = {
        "canonical_name": normalized_resolution_name(canonical_name),
        "entity_type": entity_type,
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def entity_id(*, canonical_name: str, entity_type: str) -> str:
    """§6.1 content-addressed entity ID, ``e_<16 hex>``."""
    digest = sha256(_canonical_entity_payload(
        canonical_name=canonical_name, entity_type=entity_type,
    )).hexdigest()
    return "e_" + digest[:16]


# ---------------------------------------------------------------------------
# Stage A + Stage B clustering.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageACluster:
    """One Stage A group. Membership is immutable for the duration of the
    run; Stage B merges whole clusters into target clusters and never
    splits a candidate.
    """

    entity_type: str
    normalized_name: str
    mention_ids: tuple[str, ...]

    @property
    def key(self) -> str:
        return min(self.mention_ids)


def _stage_a_clusters(mentions: list[dict]) -> tuple[list[StageACluster], int]:
    """§5.1: group eligible mentions by
    ``(entity_type, normalized_resolution_name(canonical_form))``.
    Returns the clusters and the count of duplicate mention lines that
    collapsed onto the same ``mention_id``.
    """
    buckets: dict[tuple[str, str], list[str]] = {}
    seen_ids: dict[str, int] = {}
    duplicate_lines = 0
    for m in mentions:
        canonical = m["canonical_form"]
        if not isinstance(canonical, str) or canonical == "":
            # Spec §3: empty canonical_form is an upstream defect; the
            # run must fail. Surface it as a PreflightError so the
            # caller-visible failure mode matches the rest of preflight.
            raise PreflightError(
                f"mention {m.get('mention_id')!r} has empty canonical_form"
            )
        norm = normalized_resolution_name(canonical)
        if norm == "":
            raise PreflightError(
                f"mention {m.get('mention_id')!r} normalises to empty string"
            )
        mid = m["mention_id"]
        seen_ids[mid] = seen_ids.get(mid, 0) + 1
        key = (m["entity_type"], norm)
        buckets.setdefault(key, []).append(mid)
    for mids in buckets.values():
        # Spec §2.2 — identical duplicate mention lines may be
        # de-duplicated, but the manifest records their count.
        if len(mids) != len(set(mids)):
            unique = sorted(set(mids))
            duplicate_lines += len(mids) - len(unique)
            mids[:] = unique
    out: list[StageACluster] = []
    for (entity_type, norm_name), mids in buckets.items():
        sorted_mids = tuple(sorted(mids))
        out.append(
            StageACluster(
                entity_type=entity_type,
                normalized_name=norm_name,
                mention_ids=sorted_mids,
            )
        )
    out.sort(key=lambda c: c.key)
    return out, duplicate_lines


@dataclass
class _TargetCluster:
    """A Stage B target cluster tracks one normalised name per Stage A
    cluster already absorbed, plus every ``mention_id`` it owns.
    """

    entity_type: str
    normalized_names: list[str]
    mention_ids: list[str]

    def smallest_mention_id(self) -> str:
        return min(self.mention_ids)


def _cross_score(C: StageACluster, T: "_TargetCluster") -> float:
    """§5.2 cross_score — the complete-link min over normalised names of
    the full target membership (one normalized name per Stage A cluster
    already merged into ``T``).
    """
    n = C.normalized_name
    best = float("inf")
    for tn in T.normalized_names:
        s = lev_similarity(n, tn)
        if s < best:
            best = s
    return best


@dataclass
class _ResolvedCluster:
    """A Stage B target cluster after the run completes.

    ``mention_methods`` maps each ``mention_id`` to ``(method, score)``
    where ``method`` is ``"exact_normalized"`` or ``"fuzzy"`` and the
    score is the unrounded cross_score that admitted the mention.
    ``merge_scores`` collects the per-merge unrounded scores (in
    admission order) for §5.3 weakest-link confidence.
    """

    entity_type: str
    mention_ids: list[str]
    stage_a_keys: dict[str, str]            # stage_a_key -> normalized_name
    mention_methods: dict[str, tuple[str, float]]
    merge_scores: list[float]

    def smallest_mention_id(self) -> str:
        return min(self.mention_ids)


def _stage_b(
    clusters: list[StageACluster],
) -> tuple[list[_ResolvedCluster], list[_TargetCluster]]:
    """§5.2: deterministic greedy complete-link within each entity type.

    Clusters are processed globally in ascending order of their smallest
    ``mention_id``; cross-type merges are forbidden (§5 partition). The
    candidate is never split, and established targets are never merged
    with one another as a separate operation — that is what makes the
    pass deterministic and prevents single-link transitive chaining.
    """
    # Group by type while preserving the global smallest-mention_id
    # ordering across types.
    by_type: dict[str, list[StageACluster]] = {}
    for c in clusters:
        by_type.setdefault(c.entity_type, []).append(c)

    # Explicit one-to-one association: each ``_TargetCluster`` is
    # paired with exactly one ``_ResolvedCluster``. When a candidate
    # cluster merges into an older target, this map lets us look up
    # the matching resolved cluster by target identity rather than
    # relying on insertion order (``resolved[-1]`` would update the
    # wrong cluster if there are intervening targets).
    targets: list[_TargetCluster] = []
    resolved: list[_ResolvedCluster] = []
    target_to_resolved: dict[int, _ResolvedCluster] = {}

    for entity_type, type_clusters in by_type.items():
        type_clusters.sort(key=lambda c: c.key)
        for C in type_clusters:
            best_target: _TargetCluster | None = None
            best_score = float("-inf")
            for T in targets:
                if T.entity_type != entity_type:
                    continue
                score = _cross_score(C, T)
                if score >= NORMALIZED_LEVENSHTEIN_THRESHOLD:
                    qualifies = (
                        score > best_score
                        or (
                            score == best_score
                            and best_target is not None
                            and T.smallest_mention_id()
                            < best_target.smallest_mention_id()
                        )
                    )
                    if qualifies:
                        best_score = score
                        best_target = T
            if best_target is None:
                r = _ResolvedCluster(
                    entity_type=entity_type,
                    mention_ids=list(C.mention_ids),
                    stage_a_keys={C.key: C.normalized_name},
                    mention_methods={
                        mid: ("exact_normalized", 1.0) for mid in C.mention_ids
                    },
                    merge_scores=[],
                )
                t = _TargetCluster(
                    entity_type=entity_type,
                    normalized_names=[C.normalized_name],
                    mention_ids=list(C.mention_ids),
                )
                resolved.append(r)
                targets.append(t)
                target_to_resolved[id(t)] = r
            else:
                best_target.normalized_names.append(C.normalized_name)
                for mid in C.mention_ids:
                    best_target.mention_ids.append(mid)
                # Look up the resolved cluster that owns this target
                # explicitly via the one-to-one association map.
                r = target_to_resolved[id(best_target)]
                r.mention_ids.extend(C.mention_ids)
                r.stage_a_keys[C.key] = C.normalized_name
                r.merge_scores.append(best_score)
                for mid in C.mention_ids:
                    r.mention_methods[mid] = ("fuzzy", best_score)

    return resolved, targets


# ---------------------------------------------------------------------------
# §6.3 canonical-name selection.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CanonicalPick:
    canonical_name: str
    confidence: float


def _select_canonical_name(
    *, cluster: _ResolvedCluster, candidates_by_mid: dict[str, str]
) -> _CanonicalPick:
    """§6.3 deterministic canonical-name selection.

    Three-key tie-break: (1) highest frequency of the display-normalised
    ``canonical_form`` carried by member mentions; (2) lexicographically
    smallest ``mention_id`` supporting the candidate; (3) lexicographically
    smallest normalised candidate (display value through §3).
    """
    counts: dict[str, int] = {}
    smallest_mid_supporting: dict[str, str] = {}
    for mid in cluster.mention_ids:
        canonical_display = display_normalize(candidates_by_mid[mid])
        counts[canonical_display] = counts.get(canonical_display, 0) + 1
        prev = smallest_mid_supporting.get(canonical_display)
        if prev is None or mid < prev:
            smallest_mid_supporting[canonical_display] = mid
    rows = []
    for display_value, freq in counts.items():
        rows.append(
            (
                -freq,
                smallest_mid_supporting[display_value],
                normalized_resolution_name(display_value),
                display_value,
            )
        )
    rows.sort()
    chosen = rows[0][3]
    if cluster.merge_scores:
        confidence = min(cluster.merge_scores)
    else:
        confidence = 1.0
    return _CanonicalPick(canonical_name=chosen, confidence=confidence)


# ---------------------------------------------------------------------------
# §6.4 aliases.
# ---------------------------------------------------------------------------


def _aliases(
    *,
    cluster: _ResolvedCluster,
    candidates_by_mid: dict[str, str],
    surfaces_by_mid: dict[str, str],
    canonical_name: str,
) -> list[str]:
    """§6.4 distinct display-normalised aliases, excluding ``canonical_name``.

    Sort key is ``(normalized_resolution_name(alias), alias)`` ascending.
    """
    out: set[str] = set()
    for mid in cluster.mention_ids:
        for value in (candidates_by_mid[mid], surfaces_by_mid[mid]):
            disp = display_normalize(value)
            if disp == "" or disp == canonical_name:
                continue
            out.add(disp)
    decorated = sorted(
        ((normalized_resolution_name(a), a) for a in out),
        key=lambda pair: (pair[0], pair[1]),
    )
    return [a for _n, a in decorated]


# ---------------------------------------------------------------------------
# §7 atomic JSONL / JSON writers.
# ---------------------------------------------------------------------------


def _atomic_rewrite_jsonl(path: Path, lines) -> None:
    """Atomically replace ``path`` with the supplied iterator of str lines.

    Each line MUST already include its trailing newline. We write to a
    sibling temp file and ``os.replace`` it into place; a crash mid-write
    leaves the previous artifact untouched.
    """

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line)
    os.replace(tmp, path)


def _atomic_write_manifest(path: Path, payload: dict) -> None:
    """Write the §7.3 manifest atomically with sorted keys + LF + trailing newline."""

    body = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, indent=2
    ) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)
# ---------------------------------------------------------------------------
# §2.2 mention eligibility filter.
# ---------------------------------------------------------------------------


def _select_eligible_mentions(
    mentions: list[dict],
    *,
    chunk_statuses: dict[str, dict],
    extracted_types: set[str],
    rederive_mention_id,
) -> tuple[list[dict], int]:
    """Apply §2.2 eligibility and return ``(mentions, identical_duplicate_count)``.

    Per §2.2:

    * Identical duplicate mention records — every field identical —
      may be de-duplicated by ``mention_id``; the manifest records
      their count.
    * Conflicting duplicate mention records — same ``mention_id``,
      different evidence — must FAIL the run before any output is
      replaced.

    The function delegates ``mention_id`` re-derivation to the supplied
    callable so the resolver stays decoupled from extraction code while
    still enforcing the §2.2 rule 4 contract.
    """
    eligible: list[dict] = []
    duplicates = 0
    for m in mentions:
        if not isinstance(m, dict):
            raise PreflightError("mention record must be a JSON object")
        cid = m.get("chunk_id")
        status = chunk_statuses.get(cid)
        if not status or not status.get("extraction_complete"):
            continue
        if m.get("entity_type") not in extracted_types:
            continue
        mid = m.get("mention_id")
        expected = rederive_mention_id(m)
        if expected != mid:
            raise PreflightError(
                f"mention {mid!r} does not re-derive: got {expected!r}"
            )
        eligible.append(m)

    # §2.2 duplicate-mention semantics: group by mention_id and check
    # every member is byte-identical to the first; conflicting
    # duplicates abort the run before any output is replaced.
    by_mid: dict[str, list[dict]] = {}
    for m in eligible:
        by_mid.setdefault(m["mention_id"], []).append(m)
    deduped: list[dict] = []
    for mid, group in by_mid.items():
        first = group[0]
        canonical_blob = json.dumps(
            first, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        for other in group[1:]:
            other_blob = json.dumps(
                other, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            )
            if other_blob != canonical_blob:
                raise PreflightError(
                    f"conflicting duplicate mention records share "
                    f"mention_id={mid!r}; same id with different "
                    f"evidence fails the run per §2.2"
                )
        duplicates += len(group) - 1
        deduped.append(first)

    # Deterministic ordering for downstream Stage A/B iteration.
    deduped.sort(key=lambda r: (r["entity_type"], r["mention_id"]))
    return deduped, duplicates


# ---------------------------------------------------------------------------
# Output rendering (no IO yet — just builds the records).
# ---------------------------------------------------------------------------


def _round6(x: float) -> float:
    """Round to six decimals without altering ``1.0`` etc. Used only
    when writing to the storage layer; algorithm internals keep the
    unrounded values per §4 / §5.3.
    """
    return float(f"{x:.6f}")


def _render_entities_and_resolutions(
    clusters: list[_ResolvedCluster],
    *,
    candidates_by_mid: dict[str, str],
    surfaces_by_mid: dict[str, str],
):
    """Render entities + entity_resolutions records deterministically.

    Returns three lists of strings (entity lines, resolution lines, and
    a parallel list of (cluster, entity_id) tuples for the manifest).

    ``data/entities.jsonl`` is sorted by ``(entity_type, entity_id)``
    per §7.1 / §7.3 — NOT by smallest mention_id. The two keys can
    disagree when a fuzzy merge re-orders the canonical-name
    tie-break and therefore the derived ``entity_id`` order.
    """
    # Compute every record first; sort by ``(entity_type, entity_id)``
    # only after ``entity_id`` is known.
    pending_entities: list[tuple[_ResolvedCluster, dict, list[dict]]] = []
    for cluster in sorted(clusters, key=lambda c: (c.entity_type, min(c.mention_ids))):
        pick = _select_canonical_name(
            cluster=cluster, candidates_by_mid=candidates_by_mid
        )
        aliases = _aliases(
            cluster=cluster,
            candidates_by_mid=candidates_by_mid,
            surfaces_by_mid=surfaces_by_mid,
            canonical_name=pick.canonical_name,
        )
        eid = entity_id(
            canonical_name=pick.canonical_name, entity_type=cluster.entity_type
        )
        entity_record = {
            "entity_id": eid,
            "entity_type": cluster.entity_type,
            "canonical_name": pick.canonical_name,
            "mention_ids": sorted(cluster.mention_ids),
            "aliases": aliases,
            "resolution_confidence": _round6(pick.confidence),
        }
        cluster_resolutions: list[dict] = []
        for mid in sorted(cluster.mention_ids):
            method, score = cluster.mention_methods[mid]
            cluster_resolutions.append(
                {
                    "mention_id": mid,
                    "entity_id": eid,
                    "resolution_method": method,
                    "resolution_score": _round6(score),
                }
            )
        pending_entities.append((cluster, entity_record, cluster_resolutions))

    entity_records: list[dict] = []
    resolution_records: list[dict] = []
    cluster_summaries: list[dict] = []
    # §7.3 — sort entities by (entity_type, entity_id) ascending.
    pending_entities.sort(key=lambda t: (t[1]["entity_type"], t[1]["entity_id"]))
    for _cluster, entity_record, cluster_resolutions in pending_entities:
        entity_records.append(entity_record)
        resolution_records.extend(cluster_resolutions)
        cluster_summaries.append(
            {"entity_id": entity_record["entity_id"], "entity_type": entity_record["entity_type"]}
        )
    # §7.2 — resolutions are deterministically sorted by mention_id
    # across all clusters; the cluster loop above already sorted within
    # each cluster, but the cross-cluster sort is defensive about
    # construction order.
    resolution_records.sort(key=lambda r: r["mention_id"])
    return entity_records, resolution_records, cluster_summaries


# ---------------------------------------------------------------------------
# Top-level orchestrator.
# ---------------------------------------------------------------------------

def _canonical_repo_relative(path: Path) -> str:
    """§7.3 — manifest path strings are repository-relative POSIX.

    Absolute and machine-specific temporary/default paths are
    forbidden because they break byte-identical reproducibility
    across machines. The resolver always converts to a forward-slash
    POSIX string anchored at the repo root.
    """
    abs_path = path.resolve()
    repo_root = REPO_ROOT.resolve()
    try:
        rel = abs_path.relative_to(repo_root)
    except ValueError:
        # Path lives outside the repo — surface the violation rather
        # than silently emitting a machine-specific absolute path.
        raise PreflightError(
            f"manifest path {str(path)!r} lives outside the repository "
            f"root {str(repo_root)!r}; §7.3 requires repo-relative paths"
        )
    return rel.as_posix()


def _detect_entity_id_collisions(entity_records: list[dict]) -> None:
    """§6.1 — hash-prefix collisions on unequal payloads must fail loudly.

    The ID format is ``e_<16 hex>``; a collision happens when two
    distinct ``(entity_type, normalized_canonical_name)`` pairs hash
    to the same 16-hex prefix. Per the spec, this MUST abort the run
    rather than be resolved by appending a nondeterministic suffix.
    """
    seen: dict[tuple[str, str], list[dict]] = {}
    for record in entity_records:
        key = (record["entity_type"], record["canonical_name"])
        seen.setdefault(key, []).append(record)
    # Group records by entity_id; collisions are two records with the
    # same entity_id but different (entity_type, canonical_name) keys.
    by_id: dict[str, list[dict]] = {}
    for record in entity_records:
        by_id.setdefault(record["entity_id"], []).append(record)
    for eid, records in by_id.items():
        keys = {(r["entity_type"], r["canonical_name"]) for r in records}
        if len(keys) > 1:
            sample = sorted(keys)
            raise PreflightError(
                f"hash-prefix collision on entity_id={eid!r} between "
                f"unequal payloads: {sample}"
            )


def run_resolution(
    *,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    extractions_path: Path = DEFAULT_EXTRACTIONS_PATH,
    extractions_manifest_path: Path = DEFAULT_EXTRACTIONS_MANIFEST_PATH,
    ontology_path: Path = DEFAULT_ONTOLOGY_PATH,
    entities_path: Path = DEFAULT_ENTITIES_PATH,
    entity_resolutions_path: Path = DEFAULT_ENTITY_RESOLUTIONS_PATH,
    entity_resolution_manifest_path: Path = DEFAULT_ENTITY_RESOLUTION_MANIFEST_PATH,
    rederive_mention_id=None,
) -> ResolutionResult:
    """Run the §8 deterministic two-stage resolver.

    Returns a :class:`ResolutionResult` on success. Raises
    :class:`PreflightError` (the only expected failure mode) before any
    output is written.
    """

    if rederive_mention_id is None:
        rederive_mention_id = _default_rederive_mention_id
    pre = _preflight(
        chunks_path=chunks_path,
        extractions_path=extractions_path,
        extractions_manifest_path=extractions_manifest_path,
        ontology_path=ontology_path,
    )
    ontology_version = pre["ontology_version"]
    # The preflight already validated ontology shape and version; reuse
    # the loaded ontology rather than re-parsing the file.
    ontology = _load_ontology(ontology_path)
    extracted_types = _extracted_entity_types(ontology)

    eligible, duplicates = _select_eligible_mentions(
        pre["mentions"],
        chunk_statuses=pre["chunk_statuses"],
        extracted_types=extracted_types,
        rederive_mention_id=rederive_mention_id,
    )

    stage_a, _duplicates_a = _stage_a_clusters(eligible)
    resolved, _targets = _stage_b(stage_a)

    candidates_by_mid: dict[str, str] = {m["mention_id"]: m["canonical_form"] for m in eligible}
    surfaces_by_mid: dict[str, str] = {m["mention_id"]: m["surface_form"] for m in eligible}

    entity_records, resolution_records, _summaries = _render_entities_and_resolutions(
        resolved,
        candidates_by_mid=candidates_by_mid,
        surfaces_by_mid=surfaces_by_mid,
    )

    # §6.1 — hash-prefix collisions between unequal payloads fail loudly.
    _detect_entity_id_collisions(entity_records)

    # --- Write §7.1 entities.jsonl -----------------------------------
    entity_lines = (
        json.dumps(rec, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        for rec in entity_records
    )
    _atomic_rewrite_jsonl(entities_path, entity_lines)

    # --- Write §7.2 entity_resolutions.jsonl -------------------------
    resolution_lines = (
        json.dumps(rec, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        for rec in resolution_records
    )
    _atomic_rewrite_jsonl(entity_resolutions_path, resolution_lines)

    # --- Compute SHA256s + write §7.3 manifest -----------------------
    entities_sha = _file_sha256(entities_path)
    entity_resolutions_sha = _file_sha256(entity_resolutions_path)
    chunks_sha = _file_sha256(chunks_path)
    extractions_sha = _file_sha256(extractions_path)
    extractions_manifest_sha = _file_sha256(extractions_manifest_path)
    ontology_sha = pre["ontology_sha"]

    # §7.3 — manifest path strings are repository-relative POSIX.
    manifest = {
        "algorithm": {
            "algorithm_version": ENTITY_RESOLUTION_ALGORITHM_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "similarity_metric": SIMILARITY_METRIC,
            "threshold": NORMALIZED_LEVENSHTEIN_THRESHOLD,
        },
        "counts": {
            "eligible_mentions": len(eligible),
            "entities": len(entity_records),
            "extraction_complete_chunks": sum(
                1 for st in pre["chunk_statuses"].values() if st.get("extraction_complete")
            ),
            "identical_duplicate_mentions": duplicates,
        },
        "inputs": {
            "chunks_path": _canonical_repo_relative(chunks_path),
            "chunks_sha256": chunks_sha,
            "extractions_manifest_path": _canonical_repo_relative(extractions_manifest_path),
            "extractions_manifest_sha256": extractions_manifest_sha,
            "extractions_path": _canonical_repo_relative(extractions_path),
            "extractions_sha256": extractions_sha,
            "ontology_path": _canonical_repo_relative(ontology_path),
            "ontology_sha256": ontology_sha,
            "ontology_version": ontology_version,
        },
        "outputs": {
            "entities_path": _canonical_repo_relative(entities_path),
            "entities_records": len(entity_records),
            "entities_sha256": entities_sha,
            "entity_resolutions_path": _canonical_repo_relative(entity_resolutions_path),
            "entity_resolutions_records": len(resolution_records),
            "entity_resolutions_sha256": entity_resolutions_sha,
        },
        "schema_version": 1,
    }
    # Spec §7.3: manifest written LAST as the generation's commit marker.
    _atomic_write_manifest(entity_resolution_manifest_path, manifest)

    return ResolutionResult(
        entities_records=len(entity_records),
        entity_resolutions_records=len(resolution_records),
        eligible_mentions=len(eligible),
        identical_duplicate_mentions=duplicates,
        extraction_complete_chunks=manifest["counts"]["extraction_complete_chunks"],
    )
