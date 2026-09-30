"""Generate deterministic synthetic maintenance records and retrieval eval cases."""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "seed_logs.jsonl"
EVAL_PATH = ROOT / "data" / "eval_queries.json"
SEED = 20260929
NAMESPACE = UUID("f447d2c3-4299-45c8-9af4-7a3902c976c1")
EQUIPMENT_TYPES = (
    "hydraulic pump",
    "electric motor",
    "conveyor belt",
    "compressor",
    "gearbox",
    "cooling fan",
    "valve actuator",
    "PLC controller",
)
SEVERITIES = ("low", "medium", "high", "critical")
SEVERITY_WEIGHTS = (0.15, 0.40, 0.35, 0.10)
CODE_PREFIXES = {
    "hydraulic pump": "HYD",
    "electric motor": "MOT",
    "conveyor belt": "CVB",
    "compressor": "CMP",
    "gearbox": "GBX",
    "cooling fan": "FAN",
    "valve actuator": "VLV",
    "PLC controller": "PLC",
}
OPERATING_CONTEXTS = (
    "a cold start",
    "full-load operation",
    "recovery from washdown",
    "the night shift",
    "a pressure test",
    "startup after a long idle",
    "the afternoon production run",
    "a commanded speed change",
    "the first cycle of the shift",
    "a loaded transfer",
    "a routine inspection",
    "extended operation",
    "a restart after maintenance",
    "temperature stabilization",
)
SPECIAL_RECORDS = (
    {
        "equipment": "hydraulic pump",
        "severity": "high",
        "text": (
            "The hydraulic unit shudders at start-up and leaves a narrow oil trail beneath the outlet seal. "
            "A nicked seal lip let pressure push fluid past the shaft. "
            "The mechanic replaced the seal and retested the pump at operating pressure."
        ),
    },
    {
        "equipment": "electric motor",
        "severity": "high",
        "text": (
            "The drive motor gives a low growl and slows whenever the production line is loaded. "
            "A dry bearing increased drag and pulled current above the nameplate value. "
            "The technician replaced both bearings and confirmed a stable current draw."
        ),
    },
    {
        "equipment": "conveyor belt",
        "severity": "medium",
        "text": (
            "Material drifts to the left edge and a black rubber dust line forms by the tail pulley. "
            "Uneven idler height was steering the belt off center. "
            "The crew squared the idlers and tracked the belt through three loaded cycles."
        ),
    },
    {
        "equipment": "compressor",
        "severity": "high",
        "text": (
            "Receiver pressure falls between cycles, so the compressor restarts every few minutes. "
            "A leaking check valve let stored air return through the pump head. "
            "The valve was renewed and the receiver held pressure during a leak test."
        ),
    },
    {
        "equipment": "gearbox",
        "severity": "critical",
        "text": (
            "A coarse scraping sound rises with speed, and the gearbox output shaft has visible play. "
            "Worn tapered rollers opened the gear mesh beyond its normal clearance. "
            "The team installed new bearings and verified backlash before returning the unit to service."
        ),
    },
    {
        "equipment": "cooling fan",
        "severity": "high",
        "text": (
            "The enclosure temperature climbs while the fan runs continuously, and grille airflow is weak. "
            "Dust packed between the blades and guard restricted cooling flow. "
            "The guard and impeller were cleaned, then cabinet temperature was rechecked."
        ),
    },
    {
        "equipment": "valve actuator",
        "severity": "medium",
        "text": (
            "The process valve responds several seconds after its command and stops short of the requested position. "
            "A brittle feedback coupling slipped on the actuator shaft. "
            "The coupling was replaced and both travel limits were recalibrated."
        ),
    },
    {
        "equipment": "PLC controller",
        "severity": "high",
        "text": (
            "The operator sees a false trip on input channel 4 although the field switch remains closed. "
            "A loose terminal intermittently opened the input circuit during vibration. "
            "The terminal was retorqued and the input state was monitored through a full shift."
        ),
    },
)
SCENARIOS = {
    "hydraulic pump": (
        {
            "symptoms": (
                "During {context}, the pump loses outlet pressure and the cylinder moves in a stop-start motion.",
                "During {context}, a rattling suction sound accompanies weak flow at the return line.",
            ),
            "causes": (
                "A clogged suction strainer restricted oil supply and allowed air to collect at the inlet.",
                "A cracked inlet hose admitted air whenever the pump pulled a heavy load.",
            ),
            "actions": (
                "The strainer was cleaned, the hose was replaced, and the circuit was bled before a load test.",
                "The technician sealed the inlet, topped up the reservoir, and confirmed steady pressure.",
            ),
        },
        {
            "symptoms": (
                "During {context}, the pump casing runs hot and the relief line carries a steady stream of oil.",
                "During {context}, actuator travel slows while the hydraulic unit emits a high-pitched whine.",
            ),
            "causes": (
                "A relief valve stuck partly open and sent useful flow back to the tank.",
                "Incorrect relief pressure caused continuous bypass and overheated the hydraulic fluid.",
            ),
            "actions": (
                "The valve spool was cleaned and its pressure setting was verified against the maintenance sheet.",
                "The relief cartridge was replaced and temperature was checked through a full operating cycle.",
            ),
        },
        {
            "symptoms": (
                "During {context}, the pump pulses at low speed and the discharge gauge needle swings sharply.",
                "During {context}, the hydraulic station becomes noisy after the oil level drops below its mark.",
            ),
            "causes": (
                "A worn coupling insert allowed the motor and pump shafts to move out of alignment.",
                "A damaged shaft seal let oil escape and reduced the reservoir level below the pickup.",
            ),
            "actions": (
                "The coupling was aligned and renewed, then the crew checked vibration under load.",
                "The seal was changed, the reservoir was refilled, and the station was checked for leaks.",
            ),
        },
    ),
    "electric motor": (
        {
            "symptoms": (
                "During {context}, the motor frame becomes hot and the overload relay trips before speed stabilizes.",
                "During {context}, the motor draws uneven current and the drive slows under production load.",
            ),
            "causes": (
                "A misaligned motor base transferred excess radial force into the drive-end bearing.",
                "A loose supply terminal caused voltage drop on one phase during acceleration.",
            ),
            "actions": (
                "The base was shimmed, terminals were retorqued, and phase current was measured after restart.",
                "The electrician corrected alignment and verified insulation and current balance.",
            ),
        },
        {
            "symptoms": (
                "During {context}, a rough humming noise comes from the motor end bell and vibration rises steadily.",
                "During {context}, the shaft feels gritty by hand and the motor takes longer to coast down.",
            ),
            "causes": (
                "Grease had hardened in the bearing and no longer formed a protective film on the rollers.",
                "A damaged bearing cage allowed the rotor to move slightly off center.",
            ),
            "actions": (
                "Both bearings were replaced and the technician recorded vibration after the warm-up run.",
                "The end bell was inspected, new bearings were fitted, and shaft runout was checked.",
            ),
        },
        {
            "symptoms": (
                "During {context}, the motor starts only after a second command and then stops without a mechanical load.",
                "During {context}, the drive reports an earth fault even though the motor turns freely by hand.",
            ),
            "causes": (
                "Moisture entered the terminal box through a damaged cable gland and lowered insulation resistance.",
                "The starter contactor had burned contacts that opened intermittently as it warmed.",
            ),
            "actions": (
                "The gland was renewed, the box was dried, and insulation resistance was retested before service.",
                "The contactor was replaced and the motor completed repeated starts without another trip.",
            ),
        },
    ),
    "conveyor belt": (
        {
            "symptoms": (
                "During {context}, the belt slips at the drive pulley and leaves polished patches on its underside.",
                "During {context}, product flow surges while belt speed falls below the commanded rate.",
            ),
            "causes": (
                "Oil on the pulley lagging reduced friction and prevented the belt from carrying the full load.",
                "The take-up tension had relaxed after the previous splice inspection.",
            ),
            "actions": (
                "The lagging was cleaned, take-up tension was reset, and loaded tracking was verified.",
                "The crew adjusted the take-up and confirmed stable speed with a full belt.",
            ),
        },
        {
            "symptoms": (
                "During {context}, the belt edge rubs the frame and rubber crumbs collect near the return rollers.",
                "During {context}, the belt walks toward the walkway side whenever the conveyor is loaded.",
            ),
            "causes": (
                "One return idler was skewed and pulled the belt away from the conveyor centerline.",
                "Material buildup under the loading chute pushed uneven weight onto the left edge.",
            ),
            "actions": (
                "The idlers were squared, the chute was cleared, and tracking was observed for three cycles.",
                "The crew removed buildup and aligned the return rollers before releasing the line.",
            ),
        },
        {
            "symptoms": (
                "During {context}, a splice flap lifts and a sharp slap repeats once per belt revolution.",
                "During {context}, the belt joint opens slightly as material passes over the transfer point.",
            ),
            "causes": (
                "Impact from a dropped casting damaged the splice cover and weakened the belt joint.",
                "A worn transfer-bed slider let the loaded belt sag and flex the splice repeatedly.",
            ),
            "actions": (
                "The damaged section was removed, a new splice was installed, and the transfer bed was inspected.",
                "The slider was replaced and the belt joint was repaired before a slow-speed test.",
            ),
        },
    ),
    "compressor": (
        {
            "symptoms": (
                "During {context}, the compressor cycles frequently and the receiver pressure falls while idle.",
                "During {context}, the discharge gauge rises slowly and air tools lose force at the far drop.",
            ),
            "causes": (
                "A leaking check valve allowed stored air to flow backward through the compressor head.",
                "A loose drain fitting leaked air from the receiver whenever the compressor stopped.",
            ),
            "actions": (
                "The valve was renewed and the receiver was left under pressure for a leak check.",
                "The fitting was sealed and output pressure was measured with the production tools connected.",
            ),
        },
        {
            "symptoms": (
                "During {context}, discharge temperature climbs and the compressor shuts down on thermal protection.",
                "During {context}, the aftercooler outlet stays hot and condensate carries over into the air line.",
            ),
            "causes": (
                "Dust blocked the cooler fins and reduced heat transfer from the compressed air.",
                "A failed fan belt stopped airflow across the intercooler during long runs.",
            ),
            "actions": (
                "The cooler was washed, airflow was restored, and discharge temperature was monitored under load.",
                "The belt was replaced and condensate was drained before a one-hour pressure run.",
            ),
        },
        {
            "symptoms": (
                "During {context}, the compressor knocks at startup and oil mist appears at the intake filter.",
                "During {context}, the unit builds pressure unevenly and emits a rhythmic metallic tap.",
            ),
            "causes": (
                "A worn reed valve was not sealing against the cylinder head during the compression stroke.",
                "The crankcase oil was below its safe mark after a slow leak at the drain plug.",
            ),
            "actions": (
                "The reed plate was replaced and the head bolts were torqued in the specified sequence.",
                "The leak was sealed, oil level restored, and the compressor was tested for stable pressure.",
            ),
        },
    ),
    "gearbox": (
        {
            "symptoms": (
                "During {context}, a grinding tone rises with shaft speed and the gearbox casing vibrates.",
                "During {context}, backlash at the output is larger than the previous inspection reading.",
            ),
            "causes": (
                "Pitting on the pinion teeth increased clearance and disturbed the gear contact pattern.",
                "A worn tapered bearing let the output shaft move under reversing load.",
            ),
            "actions": (
                "The bearing set was replaced and backlash was measured before the guard was refitted.",
                "The gears were inspected, damaged parts changed, and vibration rechecked under load.",
            ),
        },
        {
            "symptoms": (
                "During {context}, oil appears below the input shaft and the level falls between rounds.",
                "During {context}, the gearbox leaves a thin lubricant trail along the coupling guard.",
            ),
            "causes": (
                "A hardened input seal had lost contact with the shaft after extended high-temperature service.",
                "Excess fill level forced oil past the breather during warm operation.",
            ),
            "actions": (
                "The seal was renewed, the breather cleared, and the correct oil level restored.",
                "The crew reset the fill level and checked for leaks after a loaded run.",
            ),
        },
        {
            "symptoms": (
                "During {context}, the gearbox runs hot and the output speed fluctuates under a steady input.",
                "During {context}, a burnt-oil odor comes from the reducer and the housing temperature keeps climbing.",
            ),
            "causes": (
                "Low lubricant left the upper gear mesh without enough oil for cooling and film strength.",
                "A blocked breather raised internal pressure and restricted oil circulation.",
            ),
            "actions": (
                "The breather was cleaned, lubricant replenished, and the temperature trend checked during production.",
                "The oil was changed and the crew verified stable speed and temperature after restart.",
            ),
        },
    ),
    "cooling fan": (
        {
            "symptoms": (
                "During {context}, cabinet temperature rises and airflow at the fan grille feels weak.",
                "During {context}, the fan runs continuously but the motor enclosure remains hotter than normal.",
            ),
            "causes": (
                "Dust packed between the impeller blades and guard and reduced the available airflow.",
                "A clogged intake screen starved the fan even though the motor was turning at full speed.",
            ),
            "actions": (
                "The screen and impeller were cleaned, then cabinet temperature was checked after a full shift.",
                "The guard was cleared and airflow was confirmed at each cooling outlet.",
            ),
        },
        {
            "symptoms": (
                "During {context}, the fan produces a pulsing rattle and the guard visibly trembles.",
                "During {context}, a low-frequency hum grows louder as the cooling fan accelerates.",
            ),
            "causes": (
                "A blade had collected uneven residue and moved the impeller out of balance.",
                "The fan bearing had excessive play after its lubricant dried out.",
            ),
            "actions": (
                "The impeller was cleaned and balanced, and bearing play was measured before startup.",
                "The bearing was replaced and vibration was checked at operating speed.",
            ),
        },
        {
            "symptoms": (
                "During {context}, the fan fails to start on its first command and the enclosure alarm appears.",
                "During {context}, the fan slows intermittently although the incoming supply voltage is steady.",
            ),
            "causes": (
                "A weak run capacitor could not provide enough phase shift for reliable motor starting.",
                "A loose relay contact interrupted power when the panel warmed up.",
            ),
            "actions": (
                "The capacitor was replaced and the fan completed ten consecutive starts without an alarm.",
                "The relay terminals were retorqued and the fan was monitored through a warm operating cycle.",
            ),
        },
    ),
    "valve actuator": (
        {
            "symptoms": (
                "During {context}, the valve takes too long to reach its commanded position and then overshoots.",
                "During {context}, the actuator stops short and the process flow remains below its setpoint.",
            ),
            "causes": (
                "A slipping feedback coupling sent an incorrect position signal to the actuator controller.",
                "Low instrument-air pressure slowed the actuator and delayed its travel response.",
            ),
            "actions": (
                "The coupling was replaced, travel limits recalibrated, and response time checked at both ends.",
                "The air supply was restored and the valve was stroked through its full range.",
            ),
        },
        {
            "symptoms": (
                "During {context}, the actuator hunts around the setpoint and the valve stem moves in short bursts.",
                "During {context}, the valve position drifts after the command is removed.",
            ),
            "causes": (
                "A worn position sensor produced noisy feedback near the middle of actuator travel.",
                "Loose linkage allowed the stem to move after the actuator brake engaged.",
            ),
            "actions": (
                "The sensor was changed and the control loop was retuned using a slow full-stroke test.",
                "The linkage was tightened and position hold was confirmed at three setpoints.",
            ),
        },
        {
            "symptoms": (
                "During {context}, the actuator motor stalls near the seat and the valve does not close fully.",
                "During {context}, the limit switch changes state before the valve reaches its mechanical stop.",
            ),
            "causes": (
                "Debris raised seating torque above the actuator's configured travel limit.",
                "A shifted cam tripped the close limit before the stem completed its travel.",
            ),
            "actions": (
                "The valve trim was cleaned and the close limit was reset after a controlled stroke test.",
                "The cam was repositioned and the valve was leak-tested at its closed position.",
            ),
        },
    ),
    "PLC controller": (
        {
            "symptoms": (
                "During {context}, input channel 4 flickers even though the field switch remains closed.",
                "During {context}, the controller logs an intermittent input fault and stops the line without a real trip.",
            ),
            "causes": (
                "A loose terminal intermittently opened the input circuit when the panel vibrated.",
                "A damaged sensor cable had a broken conductor near the cabinet gland.",
            ),
            "actions": (
                "The terminal was retorqued and the input state was monitored through a full production shift.",
                "The cable was replaced and the controller input was tested through repeated switch cycles.",
            ),
        },
        {
            "symptoms": (
                "During {context}, an output remains energized after the operator releases the start command.",
                "During {context}, the PLC output LED stays on while the connected solenoid is de-energized.",
            ),
            "causes": (
                "A relay contact on the output card had welded after repeated inductive switching.",
                "A stale latch bit remained set after the sequence returned from an interrupted cycle.",
            ),
            "actions": (
                "The output card was replaced and the interlock sequence was tested from a cold start.",
                "The latch logic was corrected and each output was verified against the field device.",
            ),
        },
        {
            "symptoms": (
                "During {context}, the controller loses its remote I/O link and several values freeze at once.",
                "During {context}, communication warnings appear whenever the motor starter changes state.",
            ),
            "causes": (
                "A loose network connector caused intermittent packet loss at the remote rack.",
                "Poor cable shielding allowed switching noise to corrupt the fieldbus connection.",
            ),
            "actions": (
                "The connector was replaced, shielding bonded correctly, and communication counters monitored.",
                "The network cable was rerouted away from power conductors and the rack was retested.",
            ),
        },
    ),
}
SEMANTIC_QUERIES = (
    "vibrating pump leaking fluid around outlet",
    "electric drive growls and loses speed under load",
    "belt keeps walking sideways and shedding rubber",
    "air pressure will not stay stored and the unit keeps cycling",
    "grinding noise with a loose gearbox output shaft",
    "weak air circulation causes enclosure overheating",
    "control valve responds late and stops before full travel",
    "intermittent false PLC input trip",
)
EXACT_CODE_INDICES = (0, 6, 30, 46, 60, 76, 90, 106)
FILTER_CASES = (
    ("compressor cannot maintain receiver pressure", "compressor"),
    ("hydraulic pump loses prime under load", "hydraulic pump"),
    ("motor overload during production", "electric motor"),
    ("PLC input faults at high severity", "PLC controller"),
)


def _generate_template_record(
    equipment: str,
    variation: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Create one three-sentence observation from a matched cause/action scenario."""
    scenario = rng.choice(SCENARIOS[equipment])
    context = OPERATING_CONTEXTS[variation]
    text = " ".join(
        (
            rng.choice(scenario["symptoms"]).format(context=context),
            rng.choice(scenario["causes"]),
            rng.choice(scenario["actions"]),
        )
    )
    return {
        "text": text,
        "equipment": equipment,
        "severity": rng.choices(SEVERITIES, weights=SEVERITY_WEIGHTS, k=1)[0],
        "error_code": None,
    }


def _error_code(equipment: str, sequence: int) -> str:
    """Return a unique deterministic equipment-family diagnostic code."""
    if equipment == "PLC controller":
        return f"PLC-F{13 + sequence:02d}"
    prefix = CODE_PREFIXES[equipment]
    family_start = {
        "hydraulic pump": 2000,
        "electric motor": 3000,
        "conveyor belt": 4000,
        "compressor": 5000,
        "gearbox": 6000,
        "cooling fan": 7000,
        "valve actuator": 8000,
    }[equipment]
    return f"{prefix}-{family_start + sequence:04d}"


def _build_evaluation(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build semantic, exact-code, and equipment/severity-filtered eval cases."""
    cases: list[dict[str, Any]] = []
    for index, query in enumerate(SEMANTIC_QUERIES):
        cases.append(
            {
                "query": query,
                "type": "SEMANTIC",
                "expected_ids": [records[index]["id"]],
                "filters": {},
            }
        )
    for index in EXACT_CODE_INDICES:
        record = records[index]
        cases.append(
            {
                "query": record["error_code"],
                "type": "EXACT-CODE",
                "expected_ids": [record["id"]],
                "filters": {},
            }
        )
    for query, equipment in FILTER_CASES:
        expected_ids = [
            record["id"]
            for record in records
            if record["equipment"] == equipment
            and SEVERITIES.index(record["severity"]) >= SEVERITIES.index("high")
        ]
        cases.append(
            {
                "query": query,
                "type": "FILTERED",
                "expected_ids": expected_ids,
                "filters": {"equipment": equipment, "severity_gte": "high"},
            }
        )
    return cases


def generate_dataset() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate the fixed 120-record seed set and its 20 deterministic eval cases."""
    rng = random.Random(SEED)
    records = [{**record, "error_code": None} for record in SPECIAL_RECORDS]
    for equipment in EQUIPMENT_TYPES:
        for variation in range(14):
            records.append(_generate_template_record(equipment, variation, rng))

    if len(records) != 120:
        raise RuntimeError("the seed generator must create exactly 120 records")

    code_counts: Counter[str] = Counter()
    for index, record in enumerate(records):
        record["id"] = str(uuid5(NAMESPACE, f"maintenance-record-{index:03d}"))
        if index % 5 in (0, 1):
            equipment = record["equipment"]
            code = _error_code(equipment, code_counts[equipment])
            code_counts[equipment] += 1
            record["error_code"] = code
            record["text"] += f" The diagnostic display logged {code}."

    return records, _build_evaluation(records)


def write_dataset(
    records_path: str | Path = DATA_PATH,
    eval_path: str | Path = EVAL_PATH,
) -> tuple[int, int]:
    """Write JSONL seed records and the eval-query JSON manifest."""
    records, cases = generate_dataset()
    records_file = Path(records_path)
    eval_file = Path(eval_path)
    records_file.parent.mkdir(parents=True, exist_ok=True)
    eval_file.parent.mkdir(parents=True, exist_ok=True)
    records_file.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    eval_file.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(records), len(cases)


def main() -> int:
    """Generate the seed and evaluation files and print concise counts."""
    record_count, query_count = write_dataset()
    print(f"Wrote {record_count} maintenance records to {DATA_PATH}")
    print(f"Wrote {query_count} evaluation queries to {EVAL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
