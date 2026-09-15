"""Sentence-Transformers embedding adapter for BGE-small (#19).

Resolution of:

* ``docs/specs/provider-abstractions.md`` §1, §4, §7.1
* ``docs/specs/vector-retrieval.md`` §1.1 (the embedding model pin)
* ``docs/specs/implementation-spec.md`` §10

Adapter key: ``local_sentence_transformers``.

The adapter loads ``BAAI/bge-small-en-v1.5`` at the **literal pinned
git revision SHA** ``5c38ec7c405ec4b44b94cc5a9bb96e735b38267a``. The
revision is hard-coded as a module constant so a future baseline
change requires a deliberate code edit (and re-embedding the corpus);
no floating ``main`` / ``latest`` is ever requested.

The adapter L2-normalizes vectors before returning them, so
``cosine_distance`` reduces to ``1 - inner_product`` (vector-retrieval.md
§1.2). It does **not** apply the BGE query instruction; that
asymmetry is the call site's responsibility (vector-retrieval.md §1.1
"applied to queries only"). The ``#10`` resolver confirmed that
queries get the instruction, passages do not, and that the adapter
itself stays instruction-agnostic.

Reported model identity is the literal string
``local_sentence_transformers:BAAI/bge-small-en-v1.5:5c38ec7c405ec4b44b94cc5a9bb96e735b38267a``
which is what ``validate_vector_store`` (#26) will compare against
every pgvector row.

Token-counting seam (``count_tokens`` / ``count_tokens_batch``): the
adapter loads the same pinned tokenizer that the model ships with,
via HuggingFace ``AutoTokenizer.from_pretrained(..., revision=...)``.
This is the exact-tokenizer source required by
``vector-retrieval.md`` §1.3 for the last-line oversize guard. The
``tokenizer_loader`` constructor argument lets tests inject a
deterministic stand-in; production uses :func:`_default_tokenizer_loader`,
which imports ``transformers`` lazily so module import never pays the
torch / transformers cost for tests that do not exercise the seam.
"""

from __future__ import annotations

import math
from typing import Any

from src.providers.cache import ProviderCache
from src.providers.protocol import EmbeddingProvider, Usage


PROVIDER_KEY = "local_sentence_transformers"
NAME = "local_sentence_transformers"

MODEL_ID = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DIM = 384

# Full identity string persisted alongside pgvector rows by #26. Must
# match the literal expected by ``validate_vector_store`` exactly.
MODEL_IDENTITY = f"{NAME}:{MODEL_ID}:{REVISION}"

# Cache namespace; bump on adapter changes that affect output shape.
CACHE_FORMAT_VERSION = "st-bge-small-1"

# Default batch size; spec §3.2 default 64. The adapter does not batch
# internally for #19 — the call site owns batching.
REQUIRED_SECRETS: tuple[str, ...] = ()


class SentenceTransformersEmbedding(EmbeddingProvider):
    """Adapter for the pinned BGE-small-en-v1.5 sentence-transformers model.

    Constructed via the registry. ``loader`` and ``tokenizer_loader``
    are injectable so tests can verify the exact ``revision=REVISION``
    argument without performing a network download; the production
    paths use :func:`_default_loader` and
    :func:`_default_tokenizer_loader`, which import
    ``sentence_transformers`` / ``transformers`` at call time so
    module import never touches the network.
    """

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        revision: str = REVISION,
        batch_size: int = 64,
        cache: ProviderCache | None = None,
        loader: Any = None,
        tokenizer_loader: Any = None,
    ) -> None:
        if model_id != MODEL_ID:
            raise ValueError(
                f"embedding adapter is pinned to {MODEL_ID!r}; "
                f"configured EMBEDDING_MODEL={model_id!r} does not match"
            )
        if revision != REVISION:
            # Defensive: the registry passes the pinned revision. Any
            # future change must flow through a deliberate code edit
            # to this module.
            raise ValueError(
                f"embedding revision must equal {REVISION!r}; got {revision!r}"
            )
        self.name = NAME
        self.model = MODEL_ID
        self.dim = DIM
        self._revision = REVISION
        self._batch_size = batch_size
        self._cache = cache
        self._loader = loader if loader is not None else _default_loader
        self._tokenizer_loader = (
            tokenizer_loader if tokenizer_loader is not None
            else _default_tokenizer_loader
        )
        self._model = self._loader(MODEL_ID, revision=REVISION)
        self._tokenizer = self._tokenizer_loader(MODEL_ID, revision=REVISION)
        self._model_identity = MODEL_IDENTITY

    @property
    def model_identity(self) -> str:
        """The literal ``provider:model:revision`` identity string.

        Surfaced so callers (#26's pgvector loader, audit logs) can
        record the exact provenance of every embedding without
        reconstructing it from three attributes.
        """
        return self._model_identity

    def embed(self, texts: list[str]) -> tuple[list[list[float]], Usage]:
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise TypeError("embed(texts) requires list[str]")

        if self._cache is not None:
            cached = self._cache.get_embed(
                provider_name=self.name,
                provider_model=self.model,
                texts=texts,
            )
            if cached is not None:
                return cached, Usage()

        raw = self._encode(texts)
        vectors = [_l2_normalize(v) for v in raw]

        if self._cache is not None:
            self._cache.put_embed(
                provider_name=self.name,
                provider_model=self.model,
                texts=texts,
                vectors=vectors,
            )

        # Token accounting: sentence-transformers does not surface
        # exact tokenizer usage in the cheap encode() path used here.
        # Spec §6.1 lets adapters fall through with ``None``; the
        # budget layer therefore accumulates no token count for an
        # embed call. The adapter does not estimate.
        return vectors, Usage()

    # --- internals --------------------------------------------------

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """Run the model and pull out a Python ``list[list[float]]``.

        Sentence-transformers' ``encode`` returns a numpy array by
        default. We do not import numpy at the adapter module top
        level to keep the import cost small; the encode path is the
        only place numpy appears, and it is already implicit in
        sentence-transformers.
        """
        raw = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalize below explicitly
            show_progress_bar=False,
        )
        return [list(map(float, row)) for row in raw]

    # --- token-counting seam (vector-retrieval §1.3 last-line guard) -

    def count_tokens(self, text: str) -> int:
        """Return the real input-token count for ``text`` using the
        tokenizer shipped with the pinned BGE revision.

        Used by ``src.build_vector.preflight`` as the last-line
        oversize guard (vector-retrieval.md §1.3). Counts include
        special tokens (BOS / EOS) the tokenizer inserts by default;
        we do NOT pre-truncate the text — passing the full chunk
        string is the spec-required behavior.

        ``tokenizer_loader`` injection lets tests fake the count
        deterministically; production loads the real HF tokenizer
        via :func:`_default_tokenizer_loader`.
        """
        if not isinstance(text, str):
            raise TypeError("count_tokens(text) requires str")
        ids = self._tokenizer.encode(text, add_special_tokens=True)
        return len(ids)

    def count_tokens_batch(self, texts: list[str]) -> list[int]:
        """Batch variant: one real-tokenizer count per input string.

        Implemented as a sequential loop because the tokenizer object
        is stateful and parallel encoding changes the count semantics
        for some HuggingFace tokenizers; serial is correct, and the
        corpus is ~10^4 chunks where serial is cheap.
        """
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise TypeError("count_tokens_batch(texts) requires list[str]")
        return [self.count_tokens(t) for t in texts]


# --- helpers --------------------------------------------------------


def _default_loader(model_id: str, *, revision: str):
    """Late-bound ``SentenceTransformer`` loader.

    Importing ``sentence_transformers`` at module top would force a
    torch / transformer dependency on every importer of ``src.providers``,
    which is wasted work for tests that mock the loader. We import
    here so the network and torch happen only when a real adapter is
    constructed.
    """
    if revision != REVISION:
        raise ValueError(
            f"refusing to load model without pinned revision "
            f"{REVISION!r}; got revision={revision!r}"
        )
    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer(model_id, revision=revision)


def _default_tokenizer_loader(model_id: str, *, revision: str):
    """Late-bound HuggingFace tokenizer loader.

    Loads ONLY the tokenizer (no model weights, no sentence-transformers
    wrapper) so the last-line oversize guard can token-count without
    paying the torch / model-download cost. The ``revision`` argument
    pins the tokenizer to the same git SHA as the embedding model;
    a baseline-change ticket that bumps the model revision MUST bump
    this loader in lockstep (the loader refuses any non-pinned
    revision).
    """
    if revision != REVISION:
        raise ValueError(
            f"refusing to load tokenizer without pinned revision "
            f"{REVISION!r}; got revision={revision!r}"
        )
    from transformers import AutoTokenizer  # type: ignore

    return AutoTokenizer.from_pretrained(model_id, revision=revision)


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector in-place semantically; return new list.

    A zero vector stays zero (no division-by-zero). The 384-dimensional
    BGE vectors produced by the pinned model are never exactly zero
    in practice for non-empty inputs; the guard handles empty inputs
    defensively.
    """
    norm_sq = sum(x * x for x in vec)
    if norm_sq == 0:
        return [0.0 for _ in vec]
    inv = 1.0 / math.sqrt(norm_sq)
    return [x * inv for x in vec]