#!/usr/bin/env python3
"""Generate 30 text-based accident scenario JSON files for evaluation.

6 categories x 5 scenarios each. Each scenario has a German narrative
and structured ground truth for the RAG vs No-RAG comparison study.
"""
from __future__ import annotations

import json
from pathlib import Path

SCENARIOS = [
    # ── rear_end (5) ──────────────────────────────────────────────────
    {
        "scenario_id": "rear_end_001",
        "category": "rear_end",
        "scenario_text": (
            "Am 15. März 2024 gegen 08:15 Uhr ereignete sich auf der B27 in Richtung Stuttgart "
            "ein Auffahrunfall. Der PKW (Fahrzeug A), ein VW Golf, fuhr mit ca. 80 km/h auf der "
            "rechten Fahrspur. Der vorausfahrende LKW (Fahrzeug B) musste verkehrsbedingt abbremsen. "
            "Der Fahrer von Fahrzeug A bemerkte das Bremsmanöver zu spät und fuhr trotz Vollbremsung "
            "auf das Heck des LKW auf. Der Sicherheitsabstand betrug zum Zeitpunkt des Bremsmanövers "
            "nur ca. 15 Meter bei einer Geschwindigkeit von 80 km/h, was deutlich unter dem erforderlichen "
            "Mindestabstand lag. Die Straße war trocken, die Sichtverhältnisse waren gut. Der Fahrer "
            "von Fahrzeug A gab an, kurz auf sein Mobiltelefon geschaut zu haben. Am Fahrzeug A "
            "entstand erheblicher Frontschaden, am LKW leichter Heckschaden. Beide Fahrer blieben unverletzt."
        ),
        "metadata": {"location_type": "suburban", "weather": "clear", "time_of_day": "morning", "road_type": "bundesstrasse", "vehicles_involved": 2},
        "ground_truth": {
            "primary_cause": "Unzureichender Sicherheitsabstand",
            "primary_cause_taxonomy_id": "following_distance",
            "accident_type": "rear_end",
            "contributing_factors": [
                {"factor": "Zu geringer Sicherheitsabstand", "category": "human_error", "severity": "primary"},
                {"factor": "Ablenkung durch Mobiltelefon", "category": "human_error", "severity": "secondary"},
                {"factor": "Verkehrsbedingtes Abbremsen des LKW", "category": "traffic", "severity": "minor"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 100}, {"party": "Fahrzeug B", "percentage": 0}],
            "relevant_stvo": ["§4 StVO", "§23 StVO"],
            "expected_claims": [
                "Der Sicherheitsabstand war unzureichend.",
                "Der Fahrer war durch ein Mobiltelefon abgelenkt.",
                "Fahrzeug A fuhr auf das Heck des LKW auf.",
                "Der Anscheinsbeweis spricht gegen den Auffahrenden.",
            ],
            "legal_references": ["§4 StVO", "§23 StVO", "§7 StVG"],
        },
    },
    {
        "scenario_id": "rear_end_002",
        "category": "rear_end",
        "scenario_text": (
            "Am 22. Januar 2024 um 17:30 Uhr kam es auf der A8 bei Pforzheim zu einem Auffahrunfall "
            "im Berufsverkehr. Bei stockendem Verkehr bremste Fahrzeug B (Mercedes C-Klasse) ab. "
            "Fahrzeug A (BMW 3er) konnte nicht rechtzeitig bremsen und fuhr mit geschätzten 30 km/h "
            "auf. Der Fahrer von Fahrzeug A gab an, von der Sonne geblendet worden zu sein. "
            "Die Geschwindigkeit war an den Verkehrsfluss angepasst, jedoch war der Abstand zu gering. "
            "Es entstand Sachschaden an beiden Fahrzeugen. Der Beifahrer in Fahrzeug B klagte über Nackenschmerzen."
        ),
        "metadata": {"location_type": "highway", "weather": "clear", "time_of_day": "afternoon", "road_type": "autobahn", "vehicles_involved": 2},
        "ground_truth": {
            "primary_cause": "Unaufmerksamkeit durch Blendung",
            "primary_cause_taxonomy_id": "distraction",
            "accident_type": "rear_end",
            "contributing_factors": [
                {"factor": "Blendung durch tiefstehende Sonne", "category": "environmental", "severity": "primary"},
                {"factor": "Zu geringer Abstand im Stau", "category": "human_error", "severity": "secondary"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 85}, {"party": "Fahrzeug B", "percentage": 15}],
            "relevant_stvo": ["§4 StVO", "§3 StVO"],
            "expected_claims": [
                "Der Fahrer wurde durch Sonneneinstrahlung geblendet.",
                "Der Sicherheitsabstand war im stockenden Verkehr zu gering.",
                "Die Aufprallgeschwindigkeit betrug ca. 30 km/h.",
            ],
            "legal_references": ["§4 StVO", "§3 Abs. 1 StVO"],
        },
    },
    {
        "scenario_id": "rear_end_003",
        "category": "rear_end",
        "scenario_text": (
            "Am 5. November 2024 gegen 06:45 Uhr ereignete sich auf der L1100 ein Auffahrunfall bei Nebel. "
            "Fahrzeug A (Audi A4) fuhr mit ca. 70 km/h, obwohl die Sichtweite nur ca. 50 Meter betrug. "
            "Fahrzeug B (Ford Focus) stand an einer roten Ampel. Der Fahrer von A erkannte Fahrzeug B "
            "zu spät und konnte trotz Notbremsung eine Kollision nicht vermeiden. Die Aufprallgeschwindigkeit "
            "lag bei ca. 40 km/h. Die Fahrbahn war nass. Bei der Unfallaufnahme stellte die Polizei fest, "
            "dass die Bremsen von Fahrzeug A in schlechtem Zustand waren. Fahrzeug A erlitt Totalschaden."
        ),
        "metadata": {"location_type": "suburban", "weather": "fog", "time_of_day": "morning", "road_type": "landstrasse", "vehicles_involved": 2},
        "ground_truth": {
            "primary_cause": "Bremsversagen bei nicht angepasster Geschwindigkeit",
            "primary_cause_taxonomy_id": "brake_failure",
            "accident_type": "rear_end",
            "contributing_factors": [
                {"factor": "Defekte Bremsanlage", "category": "vehicle_defect", "severity": "primary"},
                {"factor": "Überhöhte Geschwindigkeit bei Nebel", "category": "human_error", "severity": "secondary"},
                {"factor": "Nasse Fahrbahn verlängerte Bremsweg", "category": "environmental", "severity": "minor"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 100}, {"party": "Fahrzeug B", "percentage": 0}],
            "relevant_stvo": ["§3 StVO", "§23 StVO"],
            "expected_claims": [
                "Die Geschwindigkeit war den Sichtverhältnissen nicht angepasst.",
                "Die Bremsanlage war mangelhaft.",
                "Die Sichtweite betrug nur ca. 50 Meter.",
                "Fahrzeug B stand bei Rot an der Ampel.",
            ],
            "legal_references": ["§3 Abs. 1 StVO", "§23 StVO", "§7 StVG"],
        },
    },
    {
        "scenario_id": "rear_end_004",
        "category": "rear_end",
        "scenario_text": (
            "Am 8. Juli 2024 gegen 14:00 Uhr kam es auf der B10 zu einem Auffahrunfall mit drei Fahrzeugen. "
            "Fahrzeug C (Opel Corsa) fuhr mit überhöhter Geschwindigkeit (ca. 100 km/h bei erlaubten 70 km/h) "
            "und konnte nicht rechtzeitig bremsen, als Fahrzeug B (Toyota Yaris) verkehrsbedingt bremste. "
            "Fahrzeug C fuhr auf B auf, wodurch B auf das davor stehende Fahrzeug A (VW Polo) geschoben wurde. "
            "Die Straße war trocken und übersichtlich. Der Fahrer von C war unerfahren und hatte den "
            "Führerschein erst seit drei Monaten."
        ),
        "metadata": {"location_type": "suburban", "weather": "clear", "time_of_day": "afternoon", "road_type": "bundesstrasse", "vehicles_involved": 3},
        "ground_truth": {
            "primary_cause": "Überhöhte Geschwindigkeit",
            "primary_cause_taxonomy_id": "speeding",
            "accident_type": "rear_end",
            "contributing_factors": [
                {"factor": "Überhöhte Geschwindigkeit", "category": "human_error", "severity": "primary"},
                {"factor": "Unzureichender Sicherheitsabstand", "category": "human_error", "severity": "secondary"},
            ],
            "responsibility": [{"party": "Fahrzeug C", "percentage": 100}, {"party": "Fahrzeug B", "percentage": 0}, {"party": "Fahrzeug A", "percentage": 0}],
            "relevant_stvo": ["§3 StVO", "§4 StVO"],
            "expected_claims": [
                "Die Geschwindigkeit lag ca. 30 km/h über dem Tempolimit.",
                "Es handelt sich um einen Kettenauffahrunfall.",
                "Fahrzeug B wurde auf Fahrzeug A aufgeschoben.",
            ],
            "legal_references": ["§3 StVO", "§4 StVO", "§7 StVG"],
        },
    },
    {
        "scenario_id": "rear_end_005",
        "category": "rear_end",
        "scenario_text": (
            "Am 12. Dezember 2024 gegen 07:00 Uhr ereignete sich ein Auffahrunfall auf der A6 bei Mannheim. "
            "Es hatte in der Nacht gefroren, die Fahrbahn war mit einer dünnen Eisschicht bedeckt. "
            "Fahrzeug A (Seat Leon) fuhr mit Sommerreifen und konnte beim Bremsen nicht genügend "
            "Verzögerung aufbauen. Fahrzeug B (Renault Clio) hatte rechtzeitig gebremst und stand bereits. "
            "Der Aufprall erfolgte mit ca. 25 km/h. Die erlaubte Höchstgeschwindigkeit war aufgrund der "
            "Witterung bereits auf 80 km/h reduziert worden."
        ),
        "metadata": {"location_type": "highway", "weather": "ice", "time_of_day": "morning", "road_type": "autobahn", "vehicles_involved": 2},
        "ground_truth": {
            "primary_cause": "Nicht angepasste Bereifung bei winterlichen Straßenverhältnissen",
            "primary_cause_taxonomy_id": "weather_conditions",
            "accident_type": "rear_end",
            "contributing_factors": [
                {"factor": "Sommerreifen bei Glätte", "category": "vehicle_defect", "severity": "primary"},
                {"factor": "Eisglätte auf Fahrbahn", "category": "environmental", "severity": "secondary"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 100}, {"party": "Fahrzeug B", "percentage": 0}],
            "relevant_stvo": ["§2 StVO", "§3 StVO"],
            "expected_claims": [
                "Fahrzeug A hatte Sommerreifen bei winterlichen Verhältnissen.",
                "Die Fahrbahn war eisglatt.",
                "Der Bremsweg war durch die ungeeignete Bereifung erheblich verlängert.",
            ],
            "legal_references": ["§2 Abs. 3a StVO", "§3 StVO"],
        },
    },
    # ── side_collision (5) ────────────────────────────────────────────
    {
        "scenario_id": "side_collision_001",
        "category": "side_collision",
        "scenario_text": (
            "Am 3. Mai 2024 gegen 10:20 Uhr kam es auf der A5 bei Frankfurt zu einem Seitenkollision "
            "beim Spurwechsel. Fahrzeug A (Porsche Cayenne) wechselte von der mittleren auf die rechte "
            "Fahrspur, ohne den Schulterblick durchzuführen. Fahrzeug B (Fiat 500), das sich bereits "
            "auf der rechten Spur befand, wurde seitlich getroffen. Fahrzeug A hatte zwar geblinkt, "
            "aber den Seitenraum nicht ausreichend kontrolliert. Beide Fahrzeuge erlitten Seitenschäden. "
            "Die Fahrerin von Fahrzeug B wurde leicht verletzt."
        ),
        "metadata": {"location_type": "highway", "weather": "clear", "time_of_day": "morning", "road_type": "autobahn", "vehicles_involved": 2},
        "ground_truth": {
            "primary_cause": "Fahrstreifenwechsel ohne ausreichende Sicherung",
            "primary_cause_taxonomy_id": "lane_change",
            "accident_type": "side_collision",
            "contributing_factors": [
                {"factor": "Kein Schulterblick", "category": "human_error", "severity": "primary"},
                {"factor": "Unzureichende Spiegelkontrolle", "category": "human_error", "severity": "secondary"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 100}, {"party": "Fahrzeug B", "percentage": 0}],
            "relevant_stvo": ["§7 StVO"],
            "expected_claims": [
                "Der Spurwechsel erfolgte ohne Schulterblick.",
                "Fahrzeug B befand sich bereits auf der rechten Spur.",
                "Der Fahrer von Fahrzeug A hat den toten Winkel nicht kontrolliert.",
            ],
            "legal_references": ["§7 Abs. 5 StVO"],
        },
    },
    {
        "scenario_id": "side_collision_002",
        "category": "side_collision",
        "scenario_text": (
            "Am 18. September 2024 um 09:00 Uhr kam es an der Kreuzung Hauptstraße/Bahnhofstraße "
            "zu einer Seitenkollision. Fahrzeug A (Skoda Octavia) bog von der Hauptstraße nach rechts "
            "in die Bahnhofstraße ab. Dabei übersah der Fahrer einen geradeaus fahrenden Radfahrer "
            "(Beteiligter B), der auf dem Radweg parallel zur Bahnhofstraße fuhr. Es kam zur Kollision "
            "im Kreuzungsbereich. Der Radfahrer stürzte und verletzte sich am Arm. Am PKW entstand "
            "leichter Schaden am vorderen Kotflügel."
        ),
        "metadata": {"location_type": "urban", "weather": "clear", "time_of_day": "morning", "road_type": "innerorts", "vehicles_involved": 2},
        "ground_truth": {
            "primary_cause": "Vorfahrtsverletzung gegenüber Radfahrer beim Abbiegen",
            "primary_cause_taxonomy_id": "right_of_way",
            "accident_type": "side_collision",
            "contributing_factors": [
                {"factor": "Missachtung des Vorrangs des Radfahrers", "category": "human_error", "severity": "primary"},
                {"factor": "Fehlender Schulterblick beim Abbiegen", "category": "human_error", "severity": "secondary"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 100}, {"party": "Radfahrer B", "percentage": 0}],
            "relevant_stvo": ["§9 StVO", "§8 StVO"],
            "expected_claims": [
                "Der Fahrer übersah den Radfahrer beim Rechtsabbiegen.",
                "Der Radfahrer hatte Vorrang auf dem Radweg.",
                "Es wurde kein Schulterblick durchgeführt.",
            ],
            "legal_references": ["§9 Abs. 3 StVO"],
        },
    },
    {
        "scenario_id": "side_collision_003",
        "category": "side_collision",
        "scenario_text": (
            "Am 27. Juni 2024 gegen 18:45 Uhr wurde auf der Schillerstraße in München ein geparktes "
            "Fahrzeug (Fahrzeug B, Honda Jazz) beschädigt, als der Fahrer von Fahrzeug A (Dacia Duster) "
            "die Fahrertür öffnete, ohne auf den fließenden Verkehr zu achten. Ein vorbeifahrendes "
            "Motorrad (Beteiligter C) konnte der geöffneten Tür nicht mehr ausweichen, touchierte diese "
            "und stürzte. Der Motorradfahrer erlitt Prellungen. Die Tür von Fahrzeug A wurde stark "
            "beschädigt, das Motorrad hatte Kratzschäden."
        ),
        "metadata": {"location_type": "urban", "weather": "clear", "time_of_day": "evening", "road_type": "innerorts", "vehicles_involved": 2},
        "ground_truth": {
            "primary_cause": "Unvorsichtiges Öffnen der Fahrzeugtür",
            "primary_cause_taxonomy_id": "door_opening",
            "accident_type": "side_collision",
            "contributing_factors": [
                {"factor": "Türöffnung ohne Verkehrskontrolle", "category": "human_error", "severity": "primary"},
                {"factor": "Enger Straßenraum", "category": "environmental", "severity": "minor"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 100}, {"party": "Motorrad C", "percentage": 0}],
            "relevant_stvo": ["§14 StVO"],
            "expected_claims": [
                "Die Fahrzeugtür wurde ohne Blick in den Verkehr geöffnet.",
                "Der Motorradfahrer konnte nicht mehr ausweichen.",
                "Der Türöffner haftet für den entstandenen Schaden.",
            ],
            "legal_references": ["§14 Abs. 1 StVO"],
        },
    },
    {
        "scenario_id": "side_collision_004",
        "category": "side_collision",
        "scenario_text": (
            "Am 14. April 2024 um 16:00 Uhr kam es auf der B28 zu einem Überholunfall. "
            "Fahrzeug A (Tesla Model 3) setzte zum Überholen von Fahrzeug B (Traktor) an. "
            "Gleichzeitig scherte Fahrzeug B ohne Blinker nach links aus, um in einen Feldweg "
            "abzubiegen. Es kam zur seitlichen Kollision. Fahrzeug A war bereits auf Höhe des "
            "Traktors, als dieser ausscherte. Der Traktorfahrer gab an, den Überholer nicht "
            "bemerkt zu haben. Beide Fahrzeuge wurden beschädigt."
        ),
        "metadata": {"location_type": "rural", "weather": "clear", "time_of_day": "afternoon", "road_type": "bundesstrasse", "vehicles_involved": 2},
        "ground_truth": {
            "primary_cause": "Überholunfall durch plötzliches Ausscheren des Überholten",
            "primary_cause_taxonomy_id": "overtaking",
            "accident_type": "side_collision",
            "contributing_factors": [
                {"factor": "Abbiegen ohne Blinker", "category": "human_error", "severity": "primary"},
                {"factor": "Fehlende Rücksicht auf überholendes Fahrzeug", "category": "human_error", "severity": "secondary"},
            ],
            "responsibility": [{"party": "Fahrzeug B", "percentage": 70}, {"party": "Fahrzeug A", "percentage": 30}],
            "relevant_stvo": ["§5 StVO", "§9 StVO"],
            "expected_claims": [
                "Der Traktor bog ohne Blinker nach links ab.",
                "Fahrzeug A war bereits im Überholvorgang.",
                "Beide Parteien tragen eine Mitschuld.",
            ],
            "legal_references": ["§5 Abs. 2 StVO", "§9 Abs. 1 StVO"],
        },
    },
    {
        "scenario_id": "side_collision_005",
        "category": "side_collision",
        "scenario_text": (
            "Am 9. August 2024 gegen 11:30 Uhr kam es auf einem Supermarkt-Parkplatz zu einer "
            "Seitenkollision. Fahrzeug A (Hyundai Tucson) fuhr rückwärts aus einer Parklücke, "
            "während gleichzeitig Fahrzeug B (Citroën C3) die Parkplatzfahrgasse befuhr. "
            "Beide Fahrer hatten einander nicht gesehen. Die Kollision erfolgte seitlich. "
            "Es entstand leichter Blechschaden an beiden Fahrzeugen."
        ),
        "metadata": {"location_type": "parking", "weather": "clear", "time_of_day": "midday", "road_type": "parkplatz", "vehicles_involved": 2},
        "ground_truth": {
            "primary_cause": "Rückwärtsfahren ohne ausreichende Sicherung auf Parkplatz",
            "primary_cause_taxonomy_id": "parking",
            "accident_type": "side_collision",
            "contributing_factors": [
                {"factor": "Rückwärtsfahren ohne Sicht", "category": "human_error", "severity": "primary"},
                {"factor": "Unaufmerksamkeit des durchfahrenden Fahrzeugs", "category": "human_error", "severity": "secondary"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 60}, {"party": "Fahrzeug B", "percentage": 40}],
            "relevant_stvo": ["§9 StVO", "§1 StVO"],
            "expected_claims": [
                "Fahrzeug A fuhr rückwärts aus der Parklücke.",
                "Beide Fahrer haben einander nicht gesehen.",
                "Auf Parkplätzen gilt erhöhte Sorgfaltspflicht.",
            ],
            "legal_references": ["§9 Abs. 5 StVO", "§1 StVO"],
        },
    },
    # ── head_on (5) ───────────────────────────────────────────────────
    {
        "scenario_id": "head_on_001",
        "category": "head_on",
        "scenario_text": (
            "Am 20. Februar 2024 gegen 02:30 Uhr kam es auf der B3 bei Heidelberg zu einem "
            "Frontalzusammenstoß. Fahrzeug A (BMW 5er) war als Geisterfahrer auf der falschen "
            "Fahrbahnseite unterwegs. Fahrzeug B (Peugeot 308) kam dem Geisterfahrer entgegen. "
            "Trotz Ausweichversuch von Fahrzeug B kam es zur frontalen Kollision. Die Polizei "
            "stellte fest, dass der Fahrer von Fahrzeug A mit 1,8 Promille stark alkoholisiert war. "
            "Beide Fahrer wurden schwer verletzt. An beiden Fahrzeugen entstand Totalschaden."
        ),
        "metadata": {"location_type": "suburban", "weather": "clear", "time_of_day": "night", "road_type": "bundesstrasse", "vehicles_involved": 2},
        "ground_truth": {
            "primary_cause": "Falschfahrt unter Alkoholeinfluss",
            "primary_cause_taxonomy_id": "wrong_way",
            "accident_type": "head_on",
            "contributing_factors": [
                {"factor": "Falschfahrt (Geisterfahrer)", "category": "human_error", "severity": "primary"},
                {"factor": "Alkoholisierung (1,8 Promille)", "category": "human_error", "severity": "primary"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 100}, {"party": "Fahrzeug B", "percentage": 0}],
            "relevant_stvo": ["§2 StVO", "§24a StVG"],
            "expected_claims": [
                "Der Fahrer von Fahrzeug A war Geisterfahrer.",
                "Der Fahrer war mit 1,8 Promille alkoholisiert.",
                "Es kam zum Frontalzusammenstoß.",
                "Der Fahrer von Fahrzeug B versuchte auszuweichen.",
            ],
            "legal_references": ["§2 Abs. 1 StVO", "§24a StVG", "§315c StGB"],
        },
    },
    {
        "scenario_id": "head_on_002",
        "category": "head_on",
        "scenario_text": (
            "Am 30. Juni 2024 gegen 15:30 Uhr kam es auf einer Landstraße bei Freiburg zu einem "
            "Frontalzusammenstoß beim Überholen. Fahrzeug A (Audi A6) überholte trotz durchgezogener "
            "Mittellinie und unübersichtlicher Kurve eine Fahrzeugkolonne. Im Gegenverkehr kam "
            "Fahrzeug B (Mercedes E-Klasse) entgegen. Beide Fahrzeuge kollidierten frontal. "
            "Der Überholvorgang war aufgrund der Kurvenlage und der Gegenverkehrssituation "
            "höchst riskant und verboten. Beide Fahrer wurden verletzt."
        ),
        "metadata": {"location_type": "rural", "weather": "clear", "time_of_day": "afternoon", "road_type": "landstrasse", "vehicles_involved": 2},
        "ground_truth": {
            "primary_cause": "Verbotenes Überholen in unübersichtlicher Kurve",
            "primary_cause_taxonomy_id": "overtaking",
            "accident_type": "head_on",
            "contributing_factors": [
                {"factor": "Überholen trotz Verbot", "category": "human_error", "severity": "primary"},
                {"factor": "Unübersichtliche Kurvenlage", "category": "environmental", "severity": "secondary"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 100}, {"party": "Fahrzeug B", "percentage": 0}],
            "relevant_stvo": ["§5 StVO"],
            "expected_claims": [
                "Das Überholen war aufgrund der durchgezogenen Mittellinie verboten.",
                "Die Kurve war unübersichtlich.",
                "Der Gegenverkehr war nicht einsehbar.",
            ],
            "legal_references": ["§5 Abs. 2 StVO", "§5 Abs. 3 StVO"],
        },
    },
    {
        "scenario_id": "head_on_003",
        "category": "head_on",
        "scenario_text": (
            "Am 11. Oktober 2024 gegen 19:00 Uhr kam es auf der L520 zu einem Frontalzusammenstoß. "
            "Der Fahrer von Fahrzeug A (Opel Insignia) war abgelenkt, da er sein Navigationsgerät "
            "bediente, und geriet dadurch auf die Gegenfahrbahn. Fahrzeug B (Nissan Qashqai) konnte "
            "nicht mehr ausweichen. Die Kollision erfolgte mit einer kombinierten Geschwindigkeit "
            "von ca. 120 km/h. Beide Fahrzeuge erlitten Totalschaden, vier Personen wurden verletzt."
        ),
        "metadata": {"location_type": "rural", "weather": "clear", "time_of_day": "evening", "road_type": "landstrasse", "vehicles_involved": 2},
        "ground_truth": {
            "primary_cause": "Ablenkung durch Navigationssystem",
            "primary_cause_taxonomy_id": "distraction",
            "accident_type": "head_on",
            "contributing_factors": [
                {"factor": "Bedienung des Navigationsgeräts während der Fahrt", "category": "human_error", "severity": "primary"},
                {"factor": "Abkommen auf die Gegenfahrbahn", "category": "human_error", "severity": "secondary"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 100}, {"party": "Fahrzeug B", "percentage": 0}],
            "relevant_stvo": ["§23 StVO", "§2 StVO"],
            "expected_claims": [
                "Der Fahrer bediente sein Navigationsgerät während der Fahrt.",
                "Fahrzeug A geriet auf die Gegenfahrbahn.",
                "Fahrzeug B konnte nicht mehr ausweichen.",
            ],
            "legal_references": ["§23 Abs. 1a StVO", "§2 Abs. 2 StVO"],
        },
    },
    {
        "scenario_id": "head_on_004",
        "category": "head_on",
        "scenario_text": (
            "Am 24. August 2024 gegen 23:15 Uhr kam es auf der B500 im Schwarzwald zu einem "
            "schweren Frontalunfall. Der Fahrer von Fahrzeug A (VW Passat) war nach einer langen "
            "Fahrt übermüdet und schlief am Steuer ein. Das Fahrzeug geriet auf die Gegenfahrbahn "
            "und kollidierte frontal mit Fahrzeug B (Suzuki Vitara). Der Fahrer von A erlitt "
            "schwere Verletzungen. Die Fahrt hatte mehr als 10 Stunden ohne Pause gedauert."
        ),
        "metadata": {"location_type": "rural", "weather": "clear", "time_of_day": "night", "road_type": "bundesstrasse", "vehicles_involved": 2},
        "ground_truth": {
            "primary_cause": "Sekundenschlaf durch Übermüdung",
            "primary_cause_taxonomy_id": "fatigue",
            "accident_type": "head_on",
            "contributing_factors": [
                {"factor": "Sekundenschlaf am Steuer", "category": "human_error", "severity": "primary"},
                {"factor": "Über 10 Stunden Fahrt ohne Pause", "category": "human_error", "severity": "secondary"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 100}, {"party": "Fahrzeug B", "percentage": 0}],
            "relevant_stvo": ["§2 StVO", "§3 StVO"],
            "expected_claims": [
                "Der Fahrer schlief am Steuer ein.",
                "Die Fahrt dauerte über 10 Stunden ohne Pause.",
                "Fahrzeug A geriet auf die Gegenfahrbahn.",
            ],
            "legal_references": ["§2 Abs. 1 StVO", "§315c StGB"],
        },
    },
    {
        "scenario_id": "head_on_005",
        "category": "head_on",
        "scenario_text": (
            "Am 16. März 2024 gegen 21:00 Uhr kam es auf der B31 bei Friedrichshafen zu einem "
            "Frontalunfall. Fahrzeug A (Seat Ibiza) war mit 1,5 Promille alkoholisiert unterwegs "
            "und geriet in einer leichten Kurve auf die Gegenfahrbahn. Fahrzeug B (Volvo XC60) "
            "konnte trotz Vollbremsung nicht ausweichen. Die Geschwindigkeit von Fahrzeug A lag "
            "bei ca. 90 km/h bei erlaubten 70 km/h. Beide Fahrzeuge wurden stark beschädigt."
        ),
        "metadata": {"location_type": "suburban", "weather": "clear", "time_of_day": "night", "road_type": "bundesstrasse", "vehicles_involved": 2},
        "ground_truth": {
            "primary_cause": "Trunkenheitsfahrt mit Fahrbahnabweichung",
            "primary_cause_taxonomy_id": "intoxication",
            "accident_type": "head_on",
            "contributing_factors": [
                {"factor": "Alkoholisierung (1,5 Promille)", "category": "human_error", "severity": "primary"},
                {"factor": "Überhöhte Geschwindigkeit", "category": "human_error", "severity": "secondary"},
                {"factor": "Kurvige Strecke", "category": "environmental", "severity": "minor"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 100}, {"party": "Fahrzeug B", "percentage": 0}],
            "relevant_stvo": ["§24a StVG", "§3 StVO", "§2 StVO"],
            "expected_claims": [
                "Der Fahrer war mit 1,5 Promille alkoholisiert.",
                "Die Geschwindigkeit lag über dem Limit.",
                "Fahrzeug A geriet in der Kurve auf die Gegenfahrbahn.",
            ],
            "legal_references": ["§24a StVG", "§316 StGB", "§3 StVO"],
        },
    },
    # ── intersection (5) ──────────────────────────────────────────────
    {
        "scenario_id": "intersection_001",
        "category": "intersection",
        "scenario_text": (
            "Am 7. April 2024 gegen 09:45 Uhr kam es an der Kreuzung Berliner Straße / Münchner "
            "Straße zu einem Unfall durch Missachtung der Vorfahrt. Fahrzeug A (Mazda CX-5) kam "
            "aus der untergeordneten Münchner Straße und fuhr trotz Stopp-Schild in die Kreuzung "
            "ein. Fahrzeug B (Kia Sportage) hatte Vorfahrt auf der Berliner Straße und konnte "
            "nicht mehr rechtzeitig bremsen. Es kam zur Kollision im Kreuzungsbereich. "
            "Am Fahrzeug B entstand erheblicher Seitenschaden."
        ),
        "metadata": {"location_type": "urban", "weather": "clear", "time_of_day": "morning", "road_type": "kreuzung", "vehicles_involved": 2},
        "ground_truth": {
            "primary_cause": "Missachtung des Stopp-Schildes",
            "primary_cause_taxonomy_id": "right_of_way",
            "accident_type": "intersection",
            "contributing_factors": [
                {"factor": "Missachtung des Stopp-Schildes", "category": "human_error", "severity": "primary"},
                {"factor": "Einfahren in bevorrechtigte Straße", "category": "human_error", "severity": "secondary"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 100}, {"party": "Fahrzeug B", "percentage": 0}],
            "relevant_stvo": ["§8 StVO"],
            "expected_claims": [
                "Fahrzeug A missachtete das Stopp-Schild.",
                "Fahrzeug B hatte Vorfahrt.",
                "Die Kollision ereignete sich im Kreuzungsbereich.",
            ],
            "legal_references": ["§8 Abs. 1 StVO", "§8 Abs. 2 StVO"],
        },
    },
    {
        "scenario_id": "intersection_002",
        "category": "intersection",
        "scenario_text": (
            "Am 19. Mai 2024 gegen 08:30 Uhr überfuhr Fahrzeug A (Ford Kuga) eine rote Ampel "
            "an der Kreuzung Ringstraße / Gartenstraße. Der Fahrer gab an, die Ampel bei Gelb "
            "noch passieren zu wollen, doch die Ampel war bereits seit 2 Sekunden rot. "
            "Fahrzeug B (Mini Cooper), das bei Grün in die Kreuzung einfuhr, wurde seitlich "
            "getroffen. Die Dashcam von Fahrzeug B bestätigte den Rotlichtverstoß. "
            "Drei Personen wurden leicht verletzt."
        ),
        "metadata": {"location_type": "urban", "weather": "clear", "time_of_day": "morning", "road_type": "kreuzung", "vehicles_involved": 2},
        "ground_truth": {
            "primary_cause": "Rotlichtverstoß",
            "primary_cause_taxonomy_id": "red_light",
            "accident_type": "intersection",
            "contributing_factors": [
                {"factor": "Überfahren der roten Ampel", "category": "human_error", "severity": "primary"},
                {"factor": "Fehleinschätzung der Ampelphase", "category": "human_error", "severity": "secondary"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 100}, {"party": "Fahrzeug B", "percentage": 0}],
            "relevant_stvo": ["§37 StVO"],
            "expected_claims": [
                "Fahrzeug A überfuhr die rote Ampel.",
                "Die Ampel war bereits seit 2 Sekunden rot.",
                "Fahrzeug B fuhr bei Grün in die Kreuzung ein.",
                "Die Dashcam bestätigte den Rotlichtverstoß.",
            ],
            "legal_references": ["§37 Abs. 2 StVO"],
        },
    },
    {
        "scenario_id": "intersection_003",
        "category": "intersection",
        "scenario_text": (
            "Am 2. Oktober 2024 um 17:15 Uhr kam es an der Kreuzung Friedrichstraße / Schlossgasse "
            "zu einem Unfall. Fahrzeug A (Dacia Sandero) fuhr mit ca. 65 km/h in den "
            "Kreuzungsbereich ein (erlaubt waren 50 km/h). Fahrzeug B (Smart ForTwo) fuhr "
            "von rechts kommend bei Grün in die Kreuzung. Fahrzeug A konnte aufgrund der "
            "überhöhten Geschwindigkeit nicht mehr rechtzeitig bremsen. Es kam zur T-Bone-Kollision."
        ),
        "metadata": {"location_type": "urban", "weather": "clear", "time_of_day": "afternoon", "road_type": "kreuzung", "vehicles_involved": 2},
        "ground_truth": {
            "primary_cause": "Überhöhte Geschwindigkeit im Kreuzungsbereich",
            "primary_cause_taxonomy_id": "speeding",
            "accident_type": "intersection",
            "contributing_factors": [
                {"factor": "Überhöhte Geschwindigkeit (65 statt 50 km/h)", "category": "human_error", "severity": "primary"},
                {"factor": "Unzureichende Aufmerksamkeit im Kreuzungsbereich", "category": "human_error", "severity": "secondary"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 80}, {"party": "Fahrzeug B", "percentage": 20}],
            "relevant_stvo": ["§3 StVO", "§8 StVO"],
            "expected_claims": [
                "Fahrzeug A fuhr mit 65 km/h bei erlaubten 50 km/h.",
                "Die Geschwindigkeit war für den Kreuzungsbereich zu hoch.",
                "Fahrzeug B hatte Grün.",
            ],
            "legal_references": ["§3 Abs. 3 StVO", "§8 StVO"],
        },
    },
    {
        "scenario_id": "intersection_004",
        "category": "intersection",
        "scenario_text": (
            "Am 13. Juni 2024 gegen 12:00 Uhr kam es an einer unbeschilderten Kreuzung in einem "
            "Wohngebiet zu einem Unfall. Fahrzeug A (Renault Twingo) kam von links, Fahrzeug B "
            "(VW Up) von rechts. Der Fahrer von A beachtete die Rechts-vor-Links-Regel nicht "
            "und fuhr in die Kreuzung ein. Es kam zur Kollision. Der Fahrer von A gab an, "
            "abgelenkt gewesen zu sein und die Kreuzung nicht bemerkt zu haben."
        ),
        "metadata": {"location_type": "urban", "weather": "clear", "time_of_day": "midday", "road_type": "innerorts", "vehicles_involved": 2},
        "ground_truth": {
            "primary_cause": "Missachtung der Rechts-vor-Links-Regel durch Ablenkung",
            "primary_cause_taxonomy_id": "distraction",
            "accident_type": "intersection",
            "contributing_factors": [
                {"factor": "Ablenkung des Fahrers", "category": "human_error", "severity": "primary"},
                {"factor": "Missachtung Rechts-vor-Links", "category": "human_error", "severity": "primary"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 100}, {"party": "Fahrzeug B", "percentage": 0}],
            "relevant_stvo": ["§8 StVO"],
            "expected_claims": [
                "Der Fahrer von A beachtete die Rechts-vor-Links-Regel nicht.",
                "Die Kreuzung war unbeschildert.",
                "Der Fahrer war abgelenkt.",
            ],
            "legal_references": ["§8 Abs. 1 StVO"],
        },
    },
    {
        "scenario_id": "intersection_005",
        "category": "intersection",
        "scenario_text": (
            "Am 25. November 2024 gegen 16:00 Uhr wollte Fahrzeug A (Citroën Berlingo) an einer "
            "Kreuzung wenden. Dabei blockierte er die Kreuzung und wurde von Fahrzeug B (Hyundai i30), "
            "das geradeaus fuhr, seitlich erfasst. Fahrzeug A hatte keinen Vorrang zum Wenden und "
            "behinderte den fließenden Verkehr. Die Sichtverhältnisse waren durch Regen eingeschränkt."
        ),
        "metadata": {"location_type": "urban", "weather": "rain", "time_of_day": "afternoon", "road_type": "kreuzung", "vehicles_involved": 2},
        "ground_truth": {
            "primary_cause": "Unzulässiges Wenden im Kreuzungsbereich",
            "primary_cause_taxonomy_id": "u_turn",
            "accident_type": "intersection",
            "contributing_factors": [
                {"factor": "Wenden ohne Vorrang", "category": "human_error", "severity": "primary"},
                {"factor": "Blockieren der Kreuzung", "category": "human_error", "severity": "secondary"},
                {"factor": "Eingeschränkte Sicht durch Regen", "category": "environmental", "severity": "minor"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 80}, {"party": "Fahrzeug B", "percentage": 20}],
            "relevant_stvo": ["§9 StVO", "§1 StVO"],
            "expected_claims": [
                "Fahrzeug A wendete unzulässig im Kreuzungsbereich.",
                "Der fließende Verkehr wurde behindert.",
                "Die Sichtverhältnisse waren durch Regen eingeschränkt.",
            ],
            "legal_references": ["§9 Abs. 5 StVO", "§1 Abs. 2 StVO"],
        },
    },
    # ── pedestrian (5) ────────────────────────────────────────────────
    {
        "scenario_id": "pedestrian_001",
        "category": "pedestrian",
        "scenario_text": (
            "Am 10. Januar 2024 gegen 17:45 Uhr wurde ein Fußgänger auf einem Zebrastreifen "
            "in der Kaiserstraße angefahren. Fahrzeug A (Volkswagen Tiguan) näherte sich dem "
            "Fußgängerüberweg und bremste nicht, obwohl der Fußgänger bereits die Fahrbahn "
            "betreten hatte. Der Fahrer gab an, den Fußgänger wegen der Dunkelheit und des "
            "Regens nicht gesehen zu haben. Der Fußgänger trug dunkle Kleidung. Er wurde mit "
            "einer Beinverletzung ins Krankenhaus eingeliefert."
        ),
        "metadata": {"location_type": "urban", "weather": "rain", "time_of_day": "evening", "road_type": "innerorts", "vehicles_involved": 1},
        "ground_truth": {
            "primary_cause": "Nichtbeachtung des Fußgängerüberwegs",
            "primary_cause_taxonomy_id": "pedestrian_crossing",
            "accident_type": "pedestrian",
            "contributing_factors": [
                {"factor": "Missachtung des Zebrastreifens", "category": "human_error", "severity": "primary"},
                {"factor": "Schlechte Sichtverhältnisse", "category": "environmental", "severity": "secondary"},
                {"factor": "Dunkle Kleidung des Fußgängers", "category": "environmental", "severity": "minor"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 85}, {"party": "Fußgänger", "percentage": 15}],
            "relevant_stvo": ["§26 StVO"],
            "expected_claims": [
                "Der Fußgänger befand sich auf dem Zebrastreifen.",
                "Der Fahrer bremste nicht vor dem Fußgängerüberweg.",
                "Die Sichtverhältnisse waren durch Dunkelheit und Regen eingeschränkt.",
            ],
            "legal_references": ["§26 StVO", "§3 Abs. 1 StVO"],
        },
    },
    {
        "scenario_id": "pedestrian_002",
        "category": "pedestrian",
        "scenario_text": (
            "Am 6. März 2024 gegen 14:30 Uhr wurde ein Kind (8 Jahre) beim Überqueren der "
            "Schulstraße von Fahrzeug A (Audi Q3) erfasst. Das Kind rannte zwischen geparkten "
            "Autos auf die Fahrbahn. Der Fahrer fuhr mit 40 km/h in einer 30-Zone. "
            "Trotz Bremsung konnte er das Kind nicht vermeiden. Das Kind wurde am Bein verletzt. "
            "In unmittelbarer Nähe befand sich eine Grundschule mit entsprechender Beschilderung."
        ),
        "metadata": {"location_type": "urban", "weather": "clear", "time_of_day": "afternoon", "road_type": "innerorts", "vehicles_involved": 1},
        "ground_truth": {
            "primary_cause": "Überhöhte Geschwindigkeit in Schulnähe",
            "primary_cause_taxonomy_id": "speeding",
            "accident_type": "pedestrian",
            "contributing_factors": [
                {"factor": "Überhöhte Geschwindigkeit in 30-Zone", "category": "human_error", "severity": "primary"},
                {"factor": "Kind rannte unvorhergesehen auf Fahrbahn", "category": "human_error", "severity": "secondary"},
                {"factor": "Parkende Autos versperrten Sicht", "category": "environmental", "severity": "minor"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 70}, {"party": "Aufsichtsperson", "percentage": 30}],
            "relevant_stvo": ["§3 StVO", "§26 StVO"],
            "expected_claims": [
                "Der Fahrer fuhr 40 km/h in einer 30-Zone.",
                "Ein Kind rannte zwischen geparkten Autos auf die Straße.",
                "In der Nähe befindet sich eine Grundschule.",
                "Gegenüber Kindern besteht erhöhte Sorgfaltspflicht.",
            ],
            "legal_references": ["§3 Abs. 2a StVO", "§3 StVO"],
        },
    },
    {
        "scenario_id": "pedestrian_003",
        "category": "pedestrian",
        "scenario_text": (
            "Am 29. Juli 2024 gegen 01:00 Uhr wurde ein betrunkener Fußgänger (2,1 Promille) "
            "auf der Hauptstraße von Fahrzeug A (Toyota RAV4) erfasst. Der Fußgänger überquerte "
            "die Fahrbahn bei Rot, ca. 50 Meter vom nächsten Zebrastreifen entfernt. "
            "Der Fahrer konnte trotz angepasster Geschwindigkeit (45 km/h bei erlaubten 50 km/h) "
            "nicht mehr rechtzeitig bremsen. Der Fußgänger erlitt multiple Verletzungen."
        ),
        "metadata": {"location_type": "urban", "weather": "clear", "time_of_day": "night", "road_type": "innerorts", "vehicles_involved": 1},
        "ground_truth": {
            "primary_cause": "Alkoholisierter Fußgänger überquert Fahrbahn bei Rot",
            "primary_cause_taxonomy_id": "intoxication",
            "accident_type": "pedestrian",
            "contributing_factors": [
                {"factor": "Fußgänger bei Rot über Fahrbahn", "category": "human_error", "severity": "primary"},
                {"factor": "Starke Alkoholisierung des Fußgängers", "category": "human_error", "severity": "primary"},
            ],
            "responsibility": [{"party": "Fußgänger", "percentage": 75}, {"party": "Fahrzeug A", "percentage": 25}],
            "relevant_stvo": ["§25 StVO", "§37 StVO"],
            "expected_claims": [
                "Der Fußgänger überquerte bei Rot.",
                "Der Fußgänger war mit 2,1 Promille stark alkoholisiert.",
                "Der Fahrer fuhr mit angepasster Geschwindigkeit.",
                "Die Betriebsgefahr des Fahrzeugs bleibt bestehen.",
            ],
            "legal_references": ["§25 Abs. 3 StVO", "§7 StVG"],
        },
    },
    {
        "scenario_id": "pedestrian_004",
        "category": "pedestrian",
        "scenario_text": (
            "Am 17. September 2024 gegen 08:00 Uhr wurde eine ältere Dame beim Überqueren "
            "einer Fußgängerampel in der Marktstraße angefahren. Die Ampel zeigte für Fußgänger "
            "Grün. Fahrzeug A (Lieferwagen, Mercedes Sprinter) bog rechts ab und übersah die "
            "Fußgängerin im toten Winkel. Der Fahrer hatte keinen Schulterblick durchgeführt. "
            "Die Fußgängerin wurde am Hüftgelenk schwer verletzt."
        ),
        "metadata": {"location_type": "urban", "weather": "clear", "time_of_day": "morning", "road_type": "kreuzung", "vehicles_involved": 1},
        "ground_truth": {
            "primary_cause": "Abbiegefehler ohne Schulterblick",
            "primary_cause_taxonomy_id": "distraction",
            "accident_type": "pedestrian",
            "contributing_factors": [
                {"factor": "Kein Schulterblick beim Rechtsabbiegen", "category": "human_error", "severity": "primary"},
                {"factor": "Toter Winkel des Lieferwagens", "category": "vehicle_defect", "severity": "secondary"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 100}, {"party": "Fußgängerin", "percentage": 0}],
            "relevant_stvo": ["§9 StVO", "§26 StVO"],
            "expected_claims": [
                "Der Fahrer führte keinen Schulterblick durch.",
                "Die Fußgängerin hatte Grün.",
                "Der tote Winkel des Lieferwagens war ursächlich.",
            ],
            "legal_references": ["§9 Abs. 3 StVO"],
        },
    },
    {
        "scenario_id": "pedestrian_005",
        "category": "pedestrian",
        "scenario_text": (
            "Am 5. Dezember 2024 gegen 16:30 Uhr kam es in einer verkehrsberuhigten Zone "
            "(Spielstraße) zu einem Unfall. Ein Jogger (Beteiligter B) lief auf der Fahrbahn. "
            "Fahrzeug A (Range Rover) fuhr mit ca. 25 km/h, obwohl in der Spielstraße nur "
            "Schrittgeschwindigkeit (max. 7 km/h) erlaubt ist. Der Jogger wurde von hinten "
            "erfasst und stürzte. Er erlitt Schürfwunden und Prellungen."
        ),
        "metadata": {"location_type": "urban", "weather": "clear", "time_of_day": "afternoon", "road_type": "innerorts", "vehicles_involved": 1},
        "ground_truth": {
            "primary_cause": "Überhöhte Geschwindigkeit in verkehrsberuhigter Zone",
            "primary_cause_taxonomy_id": "right_of_way",
            "accident_type": "pedestrian",
            "contributing_factors": [
                {"factor": "Deutlich überhöhte Geschwindigkeit in Spielstraße", "category": "human_error", "severity": "primary"},
                {"factor": "Fußgänger auf Fahrbahn (erlaubt in Spielstraße)", "category": "traffic", "severity": "minor"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 90}, {"party": "Jogger B", "percentage": 10}],
            "relevant_stvo": ["§42 StVO", "§3 StVO"],
            "expected_claims": [
                "In der Spielstraße gilt Schrittgeschwindigkeit.",
                "Das Fahrzeug fuhr mit ca. 25 km/h statt max. 7 km/h.",
                "In verkehrsberuhigten Zonen haben Fußgänger Vorrang.",
            ],
            "legal_references": ["§42 Abs. 4a StVO", "§3 StVO"],
        },
    },
    # ── single_vehicle (5) ────────────────────────────────────────────
    {
        "scenario_id": "single_vehicle_001",
        "category": "single_vehicle",
        "scenario_text": (
            "Am 1. September 2024 gegen 22:00 Uhr kam Fahrzeug A (BMW M3) auf der B462 im "
            "Kinzigtal von der Fahrbahn ab. Der Fahrer fuhr in einer langgezogenen Linkskurve "
            "mit ca. 130 km/h (erlaubt: 80 km/h). Das Fahrzeug brach aus und prallte gegen die "
            "Leitplanke. Der Fahrer blieb unverletzt, das Fahrzeug erlitt erheblichen Front- und "
            "Seitenschaden. Es waren keine anderen Fahrzeuge beteiligt."
        ),
        "metadata": {"location_type": "rural", "weather": "clear", "time_of_day": "night", "road_type": "bundesstrasse", "vehicles_involved": 1},
        "ground_truth": {
            "primary_cause": "Überhöhte Geschwindigkeit in Kurve",
            "primary_cause_taxonomy_id": "speeding",
            "accident_type": "single_vehicle",
            "contributing_factors": [
                {"factor": "Massiv überhöhte Geschwindigkeit (130 statt 80 km/h)", "category": "human_error", "severity": "primary"},
                {"factor": "Kurvenlage der Straße", "category": "environmental", "severity": "minor"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 100}, {"party": "Straßenbaulastträger", "percentage": 0}],
            "relevant_stvo": ["§3 StVO"],
            "expected_claims": [
                "Die Geschwindigkeit lag 50 km/h über dem Limit.",
                "Das Fahrzeug kam in einer Linkskurve von der Fahrbahn ab.",
                "Es waren keine anderen Fahrzeuge beteiligt.",
            ],
            "legal_references": ["§3 Abs. 1 StVO"],
        },
    },
    {
        "scenario_id": "single_vehicle_002",
        "category": "single_vehicle",
        "scenario_text": (
            "Am 15. August 2024 gegen 13:30 Uhr platzte bei Fahrzeug A (Fiat Ducato Wohnmobil) "
            "auf der A7 bei Kassel der rechte Vorderreifen. Das Fahrzeug geriet ins Schleudern "
            "und prallte gegen die Mittelleitplanke. Der Fahrer hatte den Reifendruck seit "
            "mehreren Monaten nicht kontrolliert. Das Profil war stark abgefahren. "
            "Zwei Insassen wurden leicht verletzt."
        ),
        "metadata": {"location_type": "highway", "weather": "clear", "time_of_day": "afternoon", "road_type": "autobahn", "vehicles_involved": 1},
        "ground_truth": {
            "primary_cause": "Reifenplatzer durch mangelnde Wartung",
            "primary_cause_taxonomy_id": "tire_blowout",
            "accident_type": "single_vehicle",
            "contributing_factors": [
                {"factor": "Reifenplatzer", "category": "vehicle_defect", "severity": "primary"},
                {"factor": "Vernachlässigte Reifenwartung", "category": "human_error", "severity": "secondary"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 100}, {"party": "Reifenhersteller", "percentage": 0}],
            "relevant_stvo": ["§23 StVO", "§36 StVZO"],
            "expected_claims": [
                "Der Reifen platzte aufgrund mangelnder Wartung.",
                "Der Reifendruck wurde seit Monaten nicht kontrolliert.",
                "Das Reifenprofil war stark abgefahren.",
            ],
            "legal_references": ["§23 StVO", "§36 StVZO"],
        },
    },
    {
        "scenario_id": "single_vehicle_003",
        "category": "single_vehicle",
        "scenario_text": (
            "Am 3. November 2024 gegen 06:00 Uhr kam Fahrzeug A (Volvo V60) auf der L340 bei "
            "starkem Regen und Aquaplaning von der Fahrbahn ab. Die Geschwindigkeit betrug "
            "ca. 90 km/h bei erlaubten 100 km/h. Das Fahrzeug rutschte über den Grünstreifen "
            "und kam in einem Graben zum Stehen. Die Fahrbahn wies Spurrinnen auf, in denen sich "
            "Wasser gesammelt hatte. Der Fahrer blieb unverletzt."
        ),
        "metadata": {"location_type": "rural", "weather": "rain", "time_of_day": "morning", "road_type": "landstrasse", "vehicles_involved": 1},
        "ground_truth": {
            "primary_cause": "Aquaplaning durch Nässe und Spurrinnen",
            "primary_cause_taxonomy_id": "weather_conditions",
            "accident_type": "single_vehicle",
            "contributing_factors": [
                {"factor": "Aquaplaning auf nasser Fahrbahn", "category": "environmental", "severity": "primary"},
                {"factor": "Spurrinnen mit Wasseransammlung", "category": "environmental", "severity": "secondary"},
                {"factor": "Geschwindigkeit nicht den Verhältnissen angepasst", "category": "human_error", "severity": "secondary"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 60}, {"party": "Straßenbaulastträger", "percentage": 40}],
            "relevant_stvo": ["§3 StVO"],
            "expected_claims": [
                "Das Fahrzeug geriet in Aquaplaning.",
                "Spurrinnen mit Wasseransammlungen waren vorhanden.",
                "Die Geschwindigkeit hätte den Witterungsverhältnissen angepasst werden müssen.",
            ],
            "legal_references": ["§3 Abs. 1 StVO", "§839 BGB"],
        },
    },
    {
        "scenario_id": "single_vehicle_004",
        "category": "single_vehicle",
        "scenario_text": (
            "Am 20. Oktober 2024 gegen 05:30 Uhr wich Fahrzeug A (Škoda Superb) auf der B33 "
            "einem Reh aus und kam dabei von der Fahrbahn ab. Das Fahrzeug prallte gegen einen "
            "Baum. Der Fahrer wurde eingeklemmt und musste von der Feuerwehr befreit werden. "
            "Die Strecke war als Wildwechselgebiet ausgeschildert. Der Fahrer fuhr mit ca. 80 km/h "
            "bei erlaubten 100 km/h."
        ),
        "metadata": {"location_type": "rural", "weather": "clear", "time_of_day": "morning", "road_type": "bundesstrasse", "vehicles_involved": 1},
        "ground_truth": {
            "primary_cause": "Wildunfall durch Ausweichmanöver",
            "primary_cause_taxonomy_id": "animal_crossing",
            "accident_type": "single_vehicle",
            "contributing_factors": [
                {"factor": "Wildwechsel (Reh)", "category": "environmental", "severity": "primary"},
                {"factor": "Ausweichmanöver statt Bremsung", "category": "human_error", "severity": "secondary"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 100}, {"party": "Jagdgenossenschaft", "percentage": 0}],
            "relevant_stvo": ["§3 StVO"],
            "expected_claims": [
                "Ein Reh kreuzte die Fahrbahn.",
                "Der Fahrer wich aus statt zu bremsen.",
                "Die Strecke war als Wildwechselgebiet gekennzeichnet.",
                "Das Fahrzeug prallte gegen einen Baum.",
            ],
            "legal_references": ["§3 StVO", "§29 BJagdG"],
        },
    },
    {
        "scenario_id": "single_vehicle_005",
        "category": "single_vehicle",
        "scenario_text": (
            "Am 7. Januar 2024 gegen 08:30 Uhr kam Fahrzeug A (Ford Transit) auf der B292 "
            "auf einer vereisten Brücke ins Rutschen. Die Fahrbahn war durch überfrierende Nässe "
            "glatt, obwohl der Rest der Strecke geräumt war. Das Fahrzeug drehte sich einmal und "
            "prallte seitlich gegen die Brückenmauer. Die Brücke war nicht als Gefahrenstelle "
            "gekennzeichnet. Der Fahrer fuhr mit ca. 60 km/h."
        ),
        "metadata": {"location_type": "rural", "weather": "ice", "time_of_day": "morning", "road_type": "bundesstrasse", "vehicles_involved": 1},
        "ground_truth": {
            "primary_cause": "Glatte Fahrbahn durch überfrierende Nässe auf Brücke",
            "primary_cause_taxonomy_id": "road_conditions",
            "accident_type": "single_vehicle",
            "contributing_factors": [
                {"factor": "Vereiste Brücke ohne Warnung", "category": "environmental", "severity": "primary"},
                {"factor": "Nicht angepasste Geschwindigkeit", "category": "human_error", "severity": "secondary"},
            ],
            "responsibility": [{"party": "Fahrzeug A", "percentage": 40}, {"party": "Straßenbaulastträger", "percentage": 60}],
            "relevant_stvo": ["§3 StVO"],
            "expected_claims": [
                "Die Brücke war durch überfrierende Nässe vereist.",
                "Es gab keine Warnung vor der Gefahrenstelle.",
                "Der Rest der Strecke war geräumt.",
                "Der Straßenbaulastträger hat seine Verkehrssicherungspflicht verletzt.",
            ],
            "legal_references": ["§3 StVO", "§839 BGB", "Art. 34 GG"],
        },
    },
]


def write_scenarios(output_dir: Path) -> None:
    """Write all scenario JSON files to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for scenario in SCENARIOS:
        filename = f"{scenario['scenario_id']}.json"
        filepath = output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(scenario, f, indent=2, ensure_ascii=False)

    print(f"Written {len(SCENARIOS)} scenarios to {output_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    scenarios_dir = project_root / "evaluation" / "dataset" / "scenarios"
    write_scenarios(scenarios_dir)
