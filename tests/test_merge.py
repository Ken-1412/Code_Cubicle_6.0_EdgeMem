import random
from copy import deepcopy

from edgemem.sync import apply_delta, merge_point, merge_states

_DEVICES = ("device-a", "device-b", "device-c")


def random_point(rng, point_id="point-1"):
    """Make one valid canonical point with independently randomized field stamps."""
    fields = {
        "text": rng.choice(("pump vibrates", "bearing is warm", "seal leaks")),
        "equipment": rng.choice(("hydraulic pump", "compressor", "gearbox")),
        "severity": rng.choice(("low", "medium", "high", "critical")),
        "error_code": rng.choice((None, "E-42", "HYD-209")),
        "deleted": rng.choice((False, False, True)),
    }
    stamps = {
        field: [float(rng.randint(1, 50)), rng.choice(_DEVICES)]
        for field in fields
    }
    if fields["deleted"]:
        newest_content = max(stamp[0] for field, stamp in stamps.items() if field != "deleted")
        stamps["deleted"] = [newest_content + 1.0, rng.choice(_DEVICES)]
    updated_at, record_device = max((stamp[0], stamp[1]) for stamp in stamps.values())
    return {
        "id": point_id,
        **fields,
        "device_id": record_device,
        "updated_at": updated_at,
        "field_ts": stamps,
    }


def point_delta(point, op="upsert"):
    """Encode a canonical point as a stamped delta accepted by apply_delta."""
    return {
        "point_id": point["id"],
        "op": op,
        "fields": {
            field: {"value": point[field], "stamp": point["field_ts"][field]}
            for field in point
            if field not in {"id", "device_id", "updated_at", "field_ts"}
        },
        "ts": point["updated_at"],
        "device_id": point["device_id"],
    }


def single_field_delta(point_id, field, value, ts, device_id, op="update"):
    return {
        "point_id": point_id,
        "op": op,
        "fields": {field: {"value": value, "stamp": [float(ts), device_id]}},
        "ts": float(ts),
        "device_id": device_id,
    }


def test_merge_is_commutative_for_200_reproducible_cases():
    """Swapping peers never changes the converged canonical point."""
    rng = random.Random(42)
    for _ in range(200):
        first = random_point(rng)
        second = random_point(rng)
        assert merge_point(first, second) == merge_point(second, first)


def test_merge_is_associative_for_200_reproducible_cases():
    """Grouping three replicas differently produces the same canonical point."""
    rng = random.Random(42)
    for _ in range(200):
        first = random_point(rng)
        second = random_point(rng)
        third = random_point(rng)
        left = merge_point(merge_point(first, second), third)
        right = merge_point(first, merge_point(second, third))
        assert left == right


def test_merge_is_idempotent_for_200_reproducible_cases():
    """Reapplying one replica's state does not change its content or timestamps."""
    rng = random.Random(42)
    for _ in range(200):
        record = random_point(rng)
        assert merge_point(record, record) == record


def test_concurrent_edits_to_different_fields_are_both_preserved():
    """Independent per-field edits survive convergence together."""
    base = {
        "id": "concurrent-fields",
        "text": "pump warm",
        "equipment": "hydraulic pump",
        "severity": "medium",
        "error_code": None,
        "deleted": False,
        "device_id": "device-a",
        "updated_at": 1.0,
        "field_ts": {
            field: [1.0, "device-a"]
            for field in ("text", "equipment", "severity", "error_code", "deleted")
        },
    }
    left = deepcopy(base)
    left.update(text="pump vibration increased", updated_at=2.0, device_id="device-a")
    left["field_ts"]["text"] = [2.0, "device-a"]
    right = deepcopy(base)
    right.update(severity="critical", updated_at=3.0, device_id="device-b")
    right["field_ts"]["severity"] = [3.0, "device-b"]

    merged = merge_point(left, right)

    assert merged["text"] == left["text"]
    assert merged["severity"] == "critical"
    assert merged["field_ts"]["text"] == [2.0, "device-a"]
    assert merged["field_ts"]["severity"] == [3.0, "device-b"]


def test_same_field_tie_breaks_by_device_id():
    """When timestamps tie, lexicographically greater device_id wins."""
    left = random_point(random.Random(2))
    right = deepcopy(left)
    left.update(severity="high", device_id="device-a", updated_at=9.0)
    right.update(severity="critical", device_id="device-b", updated_at=9.0)
    left["field_ts"]["severity"] = [9.0, "device-a"]
    right["field_ts"]["severity"] = [9.0, "device-b"]

    result = merge_point(left, right)

    assert result["severity"] == "critical"
    assert result["field_ts"]["severity"] == [9.0, "device-b"]
    assert result == merge_point(right, left)


def test_tombstone_only_wins_when_newer_than_all_content_fields():
    """A later content edit beats an older delete; a newer delete hides it."""
    base = random_point(random.Random(3))
    base.update(deleted=False, updated_at=1.0, device_id="device-a")
    base["field_ts"] = {field: [1.0, "device-a"] for field in (
        "text", "equipment", "severity", "error_code", "deleted"
    )}
    delete = deepcopy(base)
    delete.update(deleted=True, updated_at=5.0, device_id="device-b")
    delete["field_ts"]["deleted"] = [5.0, "device-b"]
    later_edit = deepcopy(base)
    later_edit.update(text="repaired after inspection", updated_at=6.0, device_id="device-c")
    later_edit["field_ts"]["text"] = [6.0, "device-c"]

    live = merge_point(delete, later_edit)
    assert live["deleted"] is False
    assert live["text"] == "repaired after inspection"
    assert live == merge_point(later_edit, delete)

    newer_delete = deepcopy(delete)
    newer_delete["updated_at"] = 7.0
    newer_delete["field_ts"]["deleted"] = [7.0, "device-b"]
    tombstoned = merge_point(live, newer_delete)
    assert tombstoned["deleted"] is True
    assert tombstoned == merge_point(newer_delete, live)


def test_merge_states_merges_by_point_id_without_mutating_inputs():
    """State maps union disjoint IDs and merge overlapping IDs purely."""
    first = random_point(random.Random(4), "shared")
    other = random_point(random.Random(5), "only-a")
    incoming = deepcopy(first)
    incoming["equipment"] = "compressor"
    incoming["field_ts"]["equipment"] = [100.0, "device-z"]
    incoming["updated_at"] = 100.0
    incoming["device_id"] = "device-z"
    state_a, state_b = {"shared": first, "only-a": other}, {"shared": incoming}
    before_a, before_b = deepcopy(state_a), deepcopy(state_b)

    merged = merge_states(state_a, state_b)

    assert set(merged) == {"shared", "only-a"}
    assert merged["shared"]["equipment"] == "compressor"
    assert state_a == before_a and state_b == before_b


def test_apply_delta_is_idempotent_and_does_not_mutate_prior_state():
    """The same serialized change can be safely replayed into a state map."""
    base = random_point(random.Random(6))
    initial = {base["id"]: base}
    before = deepcopy(initial)
    delta = single_field_delta(base["id"], "severity", "critical", 55.0, "device-c")

    once = apply_delta(initial, delta)
    twice = apply_delta(once, delta)

    assert initial == before
    assert twice == once
    assert twice[base["id"]]["severity"] == "critical"


def test_shuffled_delivery_of_six_deltas_converges_to_one_state():
    """Out-of-order upsert, updates, and tombstone delivery is deterministic."""
    point_id = "point-shuffle"
    base = {
        "id": point_id,
        "text": "pump warm",
        "equipment": "hydraulic pump",
        "severity": "medium",
        "error_code": None,
        "deleted": False,
        "device_id": "device-a",
        "updated_at": 1.0,
        "field_ts": {
            field: [1.0, "device-a"]
            for field in ("text", "equipment", "severity", "error_code", "deleted")
        },
    }
    deltas = [
        point_delta(base),
        single_field_delta(point_id, "text", "pump vibration", 2, "device-a"),
        single_field_delta(point_id, "equipment", "pump-7", 3, "device-b"),
        single_field_delta(point_id, "severity", "high", 4, "device-c"),
        single_field_delta(point_id, "deleted", True, 5, "device-b", op="delete"),
        single_field_delta(point_id, "text", "pump repaired", 6, "device-c"),
    ]
    rng = random.Random(42)
    expected = None
    for _ in range(50):
        delivery = list(deltas)
        rng.shuffle(delivery)
        state = {}
        for delta in delivery:
            state = apply_delta(state, delta)
        if expected is None:
            expected = state
        assert state == expected

    point = expected[point_id]
    assert point["text"] == "pump repaired"
    assert point["equipment"] == "pump-7"
    assert point["severity"] == "high"
    assert point["deleted"] is False
