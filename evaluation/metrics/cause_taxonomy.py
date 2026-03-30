"""Canonical cause taxonomy and keyword-based classifier."""
from __future__ import annotations

from typing import Any

# 20 canonical cause taxonomy IDs with German + English keyword lists
CAUSE_TAXONOMY: dict[str, list[str]] = {
    "following_distance": [
        "abstand", "sicherheitsabstand", "auffahren", "auffahrunfall",
        "following distance", "tailgating", "zu dicht", "hinterherfahren",
    ],
    "speeding": [
        "geschwindigkeit", "überhöhte geschwindigkeit", "zu schnell",
        "speed", "speeding", "raser", "geschwindigkeitsüberschreitung",
        "tempolimit", "höchstgeschwindigkeit",
    ],
    "right_of_way": [
        "vorfahrt", "vorrang", "rechts vor links", "right of way",
        "vorfahrtsverletzung", "vorfahrtsrecht", "vorfahrtsregel",
    ],
    "red_light": [
        "rotlicht", "rote ampel", "lichtzeichen", "red light",
        "ampel überfahren", "bei rot", "rotlichtverstoß",
    ],
    "lane_change": [
        "spurwechsel", "fahrstreifenwechsel", "lane change",
        "einfädeln", "spur gewechselt", "abbiegen ohne",
    ],
    "pedestrian_crossing": [
        "fußgänger", "zebrastreifen", "fußgängerüberweg",
        "pedestrian", "crosswalk", "überqueren", "fußgängerübergang",
    ],
    "overtaking": [
        "überholen", "überholvorgang", "überholmanöver",
        "overtaking", "passing", "gegenverkehr beim überholen",
    ],
    "distraction": [
        "ablenkung", "handy", "smartphone", "unaufmerksamkeit",
        "distraction", "abgelenkt", "mobiltelefon", "unachtsamkeit",
    ],
    "intoxication": [
        "alkohol", "trunkenheit", "promille", "berauscht",
        "intoxication", "drogen", "betäubungsmittel", "fahruntüchtig",
    ],
    "tire_blowout": [
        "reifenplatzer", "reifenpanne", "tire blowout", "reifen geplatzt",
        "reifendefekt", "plattfuß",
    ],
    "brake_failure": [
        "bremsversagen", "bremsen versagt", "brake failure",
        "bremsdefekt", "bremse ausgefallen", "bremsanlage",
    ],
    "weather_conditions": [
        "wetter", "glätte", "nässe", "nebel", "regen", "schnee", "eis",
        "weather", "aquaplaning", "sichtbehinderung", "blendung",
    ],
    "wrong_way": [
        "geisterfahrer", "falsche richtung", "wrong way",
        "entgegengesetzt", "gegenfahrbahn", "falschfahrer",
    ],
    "door_opening": [
        "türöffnung", "tür geöffnet", "dooring", "door opening",
        "autotür", "beifahrertür", "fahrertür",
    ],
    "u_turn": [
        "wenden", "u-turn", "kehrtwendung", "umkehren",
        "wendemanöver", "rückwärtsfahren",
    ],
    "parking": [
        "parken", "parkvorgang", "einparken", "ausparken",
        "parking", "parkplatz", "rangieren",
    ],
    "road_conditions": [
        "straßenzustand", "schlagloch", "road conditions",
        "fahrbahnschaden", "ölspur", "baustelle", "straßenschaden",
    ],
    "animal_crossing": [
        "wildwechsel", "tier", "animal crossing", "wild",
        "wildunfall", "reh", "wildschwein",
    ],
    "load_unsecured": [
        "ladung", "ladungssicherung", "unsecured load",
        "ladung verloren", "ungesicherte ladung", "herabfallend",
    ],
    "fatigue": [
        "müdigkeit", "sekundenschlaf", "übermüdung", "fatigue",
        "eingeschlafen", "erschöpfung", "drowsy",
    ],
}


def classify_cause(free_text: str) -> str:
    """Map free-text cause description to canonical taxonomy ID.

    Uses keyword matching against the taxonomy. Returns the best-matching
    taxonomy ID, or "unknown" if no match is found.
    """
    text_lower = free_text.lower()
    best_id = "unknown"
    best_score = 0

    for taxonomy_id, keywords in CAUSE_TAXONOMY.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_id = taxonomy_id

    return best_id


def causation_accuracy_taxonomy(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute taxonomy-based causation accuracy.

    Each prediction's primary_cause is classified into a taxonomy ID and
    compared against the ground truth's primary_cause_taxonomy_id.

    Returns:
        Dict with exact_match rate and per_category breakdown.
    """
    if not predictions or not ground_truths:
        return {"exact_match": 0.0, "per_category": {}, "n": 0}

    total = min(len(predictions), len(ground_truths))
    correct = 0
    per_item: list[bool] = []
    category_correct: dict[str, int] = {}
    category_total: dict[str, int] = {}

    for pred, gt in zip(predictions[:total], ground_truths[:total]):
        pred_cause = pred.get("primary_cause", "")
        pred_taxonomy = classify_cause(pred_cause)

        gt_taxonomy = gt.get("primary_cause_taxonomy_id", "")
        if not gt_taxonomy:
            # Fall back to classifying ground truth text too
            gt_taxonomy = classify_cause(gt.get("primary_cause", ""))

        category = gt.get("category", gt.get("accident_type", "unknown"))
        category_total[category] = category_total.get(category, 0) + 1

        match = pred_taxonomy == gt_taxonomy
        per_item.append(match)
        if match:
            correct += 1
            category_correct[category] = category_correct.get(category, 0) + 1

    per_category = {
        cat: category_correct.get(cat, 0) / count
        for cat, count in category_total.items()
        if count > 0
    }

    return {
        "exact_match": correct / total if total > 0 else 0.0,
        "per_category": per_category,
        "per_item": per_item,
        "n": total,
    }
