import json
import re
from collections import Counter
from pathlib import Path
from uuid import UUID

from scripts.make_data import generate_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "seed_logs.jsonl"
EVAL_PATH = ROOT / "data" / "eval_queries.json"
REQUIRED_FIELDS = {"id", "text", "equipment", "severity", "error_code"}
SEVERITIES = {"low", "medium", "high", "critical"}
ERROR_CODE = re.compile(r"^[A-Z0-9]+-[A-Z0-9]+$")
SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def load_records():
    """Load non-empty JSONL rows from the generated seed file."""
    return [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_seed_file_has_120_valid_unique_records():
    """Every seed line follows the requested five-field maintenance schema."""
    lines = [line for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    assert len(lines) == len(records) == 120
    assert len({record["id"] for record in records}) == 120

    for record in records:
        assert set(record) == REQUIRED_FIELDS
        UUID(record["id"])
        assert isinstance(record["text"], str) and record["text"].strip()
        assert 2 <= len(re.split(r"(?<=[.!?])\s+", record["text"].strip())) <= 4
        assert isinstance(record["equipment"], str) and record["equipment"]
        assert record["severity"] in SEVERITIES
        assert record["error_code"] is None or ERROR_CODE.fullmatch(record["error_code"])
        if record["error_code"] is not None:
            assert record["error_code"] in record["text"]


def test_equipment_diversity_and_error_code_coverage():
    """The dataset covers at least six equipment types and roughly 40% coded rows."""
    records = load_records()
    assert len({record["equipment"] for record in records}) >= 6
    coded = sum(record["error_code"] is not None for record in records)
    assert 35 <= coded / len(records) * 100 <= 45


def test_evaluation_manifest_has_all_expected_cases_and_valid_references():
    """All semantic, exact-code, and filtered query cases reference real records."""
    records = load_records()
    by_id = {record["id"]: record for record in records}
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    assert len(cases) == 20
    assert Counter(case["type"] for case in cases) == {
        "SEMANTIC": 8,
        "EXACT-CODE": 8,
        "FILTERED": 4,
    }

    for case in cases:
        assert set(case) == {"query", "type", "expected_ids", "filters"}
        assert case["query"] and case["expected_ids"]
        assert set(case["expected_ids"]) <= set(by_id)
        if case["type"] == "EXACT-CODE":
            assert len(case["expected_ids"]) == 1
            record = by_id[case["expected_ids"][0]]
            assert record["error_code"] == case["query"]
            assert case["query"] in record["text"]
        if case["type"] == "FILTERED":
            filters = case["filters"]
            for record_id in case["expected_ids"]:
                record = by_id[record_id]
                if "equipment" in filters:
                    assert record["equipment"] == filters["equipment"]
                if "severity_gte" in filters:
                    assert SEVERITY_RANK[record["severity"]] >= SEVERITY_RANK[filters["severity_gte"]]


def test_generator_is_deterministic_and_matches_checked_in_files():
    """Seeded generation is reproducible and its output matches the committed dataset."""
    first_records, first_cases = generate_dataset()
    second_records, second_cases = generate_dataset()
    assert first_records == second_records
    assert first_cases == second_cases
    assert first_records == load_records()
    assert first_cases == json.loads(EVAL_PATH.read_text(encoding="utf-8"))
