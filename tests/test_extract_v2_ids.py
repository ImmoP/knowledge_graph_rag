"""Focused acceptance tests for v2 extraction IDs and records (#51)."""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extract import ids, schema  # noqa: E402
from src.extract.validation import (  # noqa: E402
    ValidatedV2Assertion,
    ValidatedV2Mention,
)


CHUNK_ID = "c_001"
DOC_ID = "W1"
SOURCE_SHA = "a" * 64
ENTITY_TYPE = "method"
RELATION_TYPE = "PROPOSES"
CHAR_START = 7
CHAR_END = 10
SUBJECT_REF = {"kind": "document", "id": DOC_ID}
OBJECT_REF = {"kind": "mention", "id": "m_0123456789abcdef"}
EVIDENCE = {
    "char_start": CHAR_START,
    "char_end": CHAR_END,
    "source_text_sha256": SOURCE_SHA,
}


ONTOLOGY = {
    "entity_types": [
        {"name": "document", "population_mode": "structural"},
        {"name": "method", "population_mode": "extracted"},
        {"name": "dataset", "population_mode": "extracted"},
    ],
    "relation_types": [
        {
            "name": "PROPOSES",
            "from": "document",
            "to": "method",
            "population_mode": "extracted",
        },
        {
            "name": "CITES",
            "from": "document",
            "to": "document",
            "population_mode": "structural",
        },
    ],
}


def _canonical_id(prefix: str, payload: dict) -> str:
    """Reproduce extraction-contract §6.1 canonical ID derivation."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:ids.HEX_PREFIX_LEN]


def _v1_mention_id() -> str:
    return ids.mention_id(
        chunk_id=CHUNK_ID,
        source_text_sha256=SOURCE_SHA,
        entity_type=ENTITY_TYPE,
        char_start=CHAR_START,
        char_end=CHAR_END,
    )


def _v1_assertion_id() -> str:
    return ids.assertion_id(
        chunk_id=CHUNK_ID,
        source_text_sha256=SOURCE_SHA,
        relation_type=RELATION_TYPE,
        subject_ref=SUBJECT_REF,
        object_ref=OBJECT_REF,
        char_start=CHAR_START,
        char_end=CHAR_END,
    )


def _v2_mention_id() -> str:
    return ids.mention_id_v2(
        chunk_id=CHUNK_ID,
        source_text_sha256=SOURCE_SHA,
        entity_type=ENTITY_TYPE,
        char_start=CHAR_START,
        char_end=CHAR_END,
    )


def _v2_assertion_id() -> str:
    return ids.assertion_id_v2(
        chunk_id=CHUNK_ID,
        source_text_sha256=SOURCE_SHA,
        relation_type=RELATION_TYPE,
        subject_ref=SUBJECT_REF,
        object_ref=OBJECT_REF,
        char_start=CHAR_START,
        char_end=CHAR_END,
    )


def _v1_mention_record() -> dict:
    return {
        "kind": "mention",
        "mention_id": _v1_mention_id(),
        "chunk_id": CHUNK_ID,
        "doc_id": DOC_ID,
        "entity_type": ENTITY_TYPE,
        "surface_form": "DPR",
        "canonical_form": "Dense Passage Retrieval",
        "evidence": dict(EVIDENCE),
        "confidence": "stated",
    }


def _v1_assertion_record() -> dict:
    return {
        "kind": "assertion",
        "assertion_id": _v1_assertion_id(),
        "chunk_id": CHUNK_ID,
        "doc_id": DOC_ID,
        "relation_type": RELATION_TYPE,
        "subject_ref": dict(SUBJECT_REF),
        "object_ref": dict(OBJECT_REF),
        "evidence": dict(EVIDENCE),
        "confidence": "stated",
    }


def _v2_mention_record() -> dict:
    record = _v1_mention_record()
    record["schema_version"] = schema.SCHEMA_VERSION_V2
    record["mention_id"] = _v2_mention_id()
    return record


def _v2_assertion_record() -> dict:
    record = _v1_assertion_record()
    record["schema_version"] = schema.SCHEMA_VERSION_V2
    record["assertion_id"] = _v2_assertion_id()
    record["evidence_quote"] = "DPR"
    return record


class TestFrozenV1Ids(unittest.TestCase):
    """The v1 public API and payload rules remain structurally frozen."""

    def test_mention_id_reproduces_v1_documented_payload(self):
        payload = {
            "chunk_id": CHUNK_ID,
            "source_text_sha256": SOURCE_SHA,
            "entity_type": ENTITY_TYPE,
            "char_start": CHAR_START,
            "char_end": CHAR_END,
        }
        expected = _canonical_id("m_", payload)
        self.assertEqual(_v1_mention_id(), expected)
        self.assertEqual(_v1_mention_id(), _v1_mention_id())

    def test_assertion_id_reproduces_v1_documented_payload(self):
        payload = {
            "chunk_id": CHUNK_ID,
            "source_text_sha256": SOURCE_SHA,
            "relation_type": RELATION_TYPE,
            "subject_ref": dict(SUBJECT_REF),
            "object_ref": dict(OBJECT_REF),
            "char_start": CHAR_START,
            "char_end": CHAR_END,
        }
        expected = _canonical_id("a_", payload)
        self.assertEqual(_v1_assertion_id(), expected)
        self.assertEqual(_v1_assertion_id(), _v1_assertion_id())

    def test_v1_id_signatures_have_no_schema_version_dispatch(self):
        for function in (ids.mention_id, ids.assertion_id):
            self.assertNotIn("schema_version", inspect.signature(function).parameters)

    def test_v1_rederive_helpers_accept_frozen_records_without_version(self):
        mention = _v1_mention_record()
        assertion = _v1_assertion_record()
        self.assertNotIn("schema_version", mention)
        self.assertNotIn("schema_version", assertion)
        self.assertEqual(ids.rederive_mention_id(mention), mention["mention_id"])
        self.assertEqual(ids.rederive_assertion_id(assertion), assertion["assertion_id"])


class TestV2Ids(unittest.TestCase):
    """v2 identity adds exactly the schema-version discriminator."""

    def test_mention_id_v2_is_deterministic_and_discriminates_v1(self):
        v2_id = _v2_mention_id()
        self.assertEqual(v2_id, _v2_mention_id())
        self.assertNotEqual(v2_id, _v1_mention_id())

    def test_mention_id_v2_reproduces_authoritative_payload(self):
        payload = {
            "chunk_id": CHUNK_ID,
            "entity_type": ENTITY_TYPE,
            "char_start": CHAR_START,
            "char_end": CHAR_END,
            "source_text_sha256": SOURCE_SHA,
            "schema_version": schema.SCHEMA_VERSION_V2,
        }
        self.assertEqual(_v2_mention_id(), _canonical_id("m_", payload))
        self.assertEqual(
            set(payload),
            {
                "chunk_id",
                "entity_type",
                "char_start",
                "char_end",
                "source_text_sha256",
                "schema_version",
            },
        )

    def test_each_v2_mention_identity_field_changes_id(self):
        baseline = _v2_mention_id()
        changes = [
            {"chunk_id": "c_002"},
            {"entity_type": "dataset"},
            {"char_start": CHAR_START + 1},
            {"char_end": CHAR_END + 1},
            {"source_text_sha256": "b" * 64},
        ]
        for changed in changes:
            with self.subTest(changed=changed):
                self.assertNotEqual(
                    baseline,
                    ids.mention_id_v2(
                        chunk_id=changed.get("chunk_id", CHUNK_ID),
                        source_text_sha256=changed.get(
                            "source_text_sha256", SOURCE_SHA
                        ),
                        entity_type=changed.get("entity_type", ENTITY_TYPE),
                        char_start=changed.get("char_start", CHAR_START),
                        char_end=changed.get("char_end", CHAR_END),
                    ),
                )

    def test_assertion_id_v2_is_deterministic_and_discriminates_v1(self):
        v2_id = _v2_assertion_id()
        self.assertEqual(v2_id, _v2_assertion_id())
        self.assertNotEqual(v2_id, _v1_assertion_id())

    def test_assertion_id_v2_reproduces_authoritative_payload(self):
        payload = {
            "chunk_id": CHUNK_ID,
            "relation_type": RELATION_TYPE,
            "subject_ref": dict(SUBJECT_REF),
            "object_ref": dict(OBJECT_REF),
            "char_start": CHAR_START,
            "char_end": CHAR_END,
            "source_text_sha256": SOURCE_SHA,
            "schema_version": schema.SCHEMA_VERSION_V2,
        }
        self.assertEqual(_v2_assertion_id(), _canonical_id("a_", payload))
        self.assertEqual(
            set(payload),
            {
                "chunk_id",
                "relation_type",
                "subject_ref",
                "object_ref",
                "char_start",
                "char_end",
                "source_text_sha256",
                "schema_version",
            },
        )

    def test_each_v2_assertion_identity_field_changes_id(self):
        baseline = _v2_assertion_id()
        changes = [
            {"chunk_id": "c_002"},
            {"relation_type": "EVALUATES_ON"},
            {"char_start": CHAR_START + 1},
            {"char_end": CHAR_END + 1},
            {"source_text_sha256": "b" * 64},
            {"subject_ref": {"kind": "document", "id": "W2"}},
            {"object_ref": {"kind": "mention", "id": "m_other"}},
        ]
        for changed in changes:
            with self.subTest(changed=changed):
                self.assertNotEqual(
                    baseline,
                    ids.assertion_id_v2(
                        chunk_id=changed.get("chunk_id", CHUNK_ID),
                        source_text_sha256=changed.get(
                            "source_text_sha256", SOURCE_SHA
                        ),
                        relation_type=changed.get("relation_type", RELATION_TYPE),
                        subject_ref=changed.get("subject_ref", SUBJECT_REF),
                        object_ref=changed.get("object_ref", OBJECT_REF),
                        char_start=changed.get("char_start", CHAR_START),
                        char_end=changed.get("char_end", CHAR_END),
                    ),
                )


class TestV2Rederivation(unittest.TestCase):
    """v2 re-derivation never upgrades ambiguous or malformed records."""

    def test_valid_v2_records_rederive_stored_ids(self):
        mention = _v2_mention_record()
        assertion = _v2_assertion_record()
        self.assertEqual(ids.rederive_v2_mention_id(mention), mention["mention_id"])
        self.assertEqual(
            ids.rederive_v2_assertion_id(assertion), assertion["assertion_id"]
        )

    def test_rederive_helpers_reject_invalid_schema_versions(self):
        invalid_versions = [None, True, False, "2", 2.0, schema.SCHEMA_VERSION_V2 + 1]
        for helper, make_record in (
            (ids.rederive_v2_mention_id, _v2_mention_record),
            (ids.rederive_v2_assertion_id, _v2_assertion_record),
        ):
            for invalid_version in invalid_versions:
                with self.subTest(helper=helper.__name__, version=invalid_version):
                    record = make_record()
                    if invalid_version is None:
                        record.pop("schema_version")
                    else:
                        record["schema_version"] = invalid_version
                    with self.assertRaises(ValueError):
                        helper(record)


class TestV2PersistedRecords(unittest.TestCase):
    """v2 records persist the aligned evidence and no duplicated literal."""

    def test_v2_mention_record_shape_and_defensive_evidence_copy(self):
        evidence = dict(EVIDENCE)
        record = ValidatedV2Mention(
            mention_id=_v2_mention_id(),
            chunk_id=CHUNK_ID,
            doc_id=DOC_ID,
            entity_type=ENTITY_TYPE,
            surface_form="DPR",
            canonical_form="Dense Passage Retrieval",
            evidence=evidence,
            confidence="stated",
        ).to_record()
        self.assertEqual(record["schema_version"], schema.SCHEMA_VERSION_V2)
        self.assertEqual(record["surface_form"], "DPR")
        self.assertEqual(set(record["evidence"]), set(EVIDENCE))
        self.assertNotIn("surface_form", record["evidence"])
        self.assertIsNot(record["evidence"], evidence)
        evidence["char_start"] = 0
        self.assertEqual(record["evidence"]["char_start"], CHAR_START)

    def test_v2_assertion_record_shape_and_defensive_dict_copies(self):
        evidence = dict(EVIDENCE)
        subject_ref = dict(SUBJECT_REF)
        object_ref = dict(OBJECT_REF)
        record = ValidatedV2Assertion(
            assertion_id=_v2_assertion_id(),
            chunk_id=CHUNK_ID,
            doc_id=DOC_ID,
            relation_type=RELATION_TYPE,
            subject_ref=subject_ref,
            object_ref=object_ref,
            evidence_quote="DPR",
            evidence=evidence,
            confidence="stated",
        ).to_record()
        self.assertEqual(record["schema_version"], schema.SCHEMA_VERSION_V2)
        self.assertEqual(record["evidence_quote"], "DPR")
        self.assertEqual(set(record["evidence"]), set(EVIDENCE))
        self.assertNotIn("evidence_quote", record["evidence"])
        self.assertEqual(record["subject_ref"], SUBJECT_REF)
        self.assertEqual(record["object_ref"], OBJECT_REF)
        for key, source in (
            ("evidence", evidence),
            ("subject_ref", subject_ref),
            ("object_ref", object_ref),
        ):
            self.assertIsNot(record[key], source)
        evidence["char_start"] = 0
        subject_ref["id"] = "W2"
        object_ref["id"] = "m_other"
        self.assertEqual(record["evidence"]["char_start"], CHAR_START)
        self.assertEqual(record["subject_ref"], SUBJECT_REF)
        self.assertEqual(record["object_ref"], OBJECT_REF)


class TestV2ModelSchemas(unittest.TestCase):
    """v2 model schemas use ontology vocabulary without model offsets."""

    def test_entity_v2_schema_shape_and_ontology_vocabulary(self):
        mention = schema.entity_pass_v2_schema(ONTOLOGY)["properties"]["mentions"]["items"]
        properties = mention["properties"]
        self.assertEqual(
            set(mention["required"]),
            {"entity_type", "surface_form", "canonical_form", "confidence"},
        )
        self.assertNotIn("char_start", properties)
        self.assertNotIn("char_end", properties)
        self.assertEqual(
            properties["occurrence_index"], {"type": "integer", "minimum": 0}
        )
        self.assertNotIn("occurrence_index", mention["required"])
        self.assertEqual(properties["entity_type"]["enum"], ["method", "dataset"])
        self.assertEqual(
            properties["confidence"]["enum"], list(schema.CONFIDENCE_VALUES)
        )

    def test_relation_v2_schema_shape_and_ontology_vocabulary(self):
        assertion = schema.relation_pass_v2_schema(ONTOLOGY)["properties"]["assertions"]["items"]
        properties = assertion["properties"]
        self.assertEqual(
            set(assertion["required"]),
            {
                "relation_type",
                "subject_ref",
                "object_ref",
                "evidence_quote",
                "confidence",
            },
        )
        self.assertEqual(properties["evidence_quote"], {"type": "string", "minLength": 1})
        self.assertNotIn("char_start", properties)
        self.assertNotIn("char_end", properties)
        self.assertEqual(
            properties["occurrence_index"], {"type": "integer", "minimum": 0}
        )
        self.assertNotIn("occurrence_index", assertion["required"])
        for endpoint in (properties["subject_ref"], properties["object_ref"]):
            self.assertEqual(set(endpoint["required"]), {"kind", "id"})
            self.assertEqual(set(endpoint["properties"]), {"kind", "id"})
        self.assertEqual(properties["relation_type"]["enum"], ["PROPOSES"])
        self.assertEqual(
            properties["confidence"]["enum"], list(schema.CONFIDENCE_VALUES)
        )
