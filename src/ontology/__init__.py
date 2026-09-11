"""Ontology validation tooling for `ontology.yaml` (#20).

Frozen v1 vocabulary drift gate. The validator operates generically from
the supplied YAML and never duplicates the v1 vocabulary in Python
(per `docs/specs/ontology.md` §4.3). Validation-policy literals
(population-mode names, phase-specific prompt symbolic names,
reserved/forbidden entity names, example statuses) are the only Python
constants; the v1 entity/relation vocabulary lives only in `ontology.yaml`.
"""
