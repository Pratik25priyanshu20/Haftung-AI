"""All agent prompts for Haftung_AI accident analysis."""
from __future__ import annotations

# --- Causation Agent Prompts ---

CAUSATION_SYSTEM_PROMPT = """Du bist ein Unfallanalytiker für die automatisierte Unfallursachenanalyse.
Deine Aufgabe ist die Bestimmung der Unfallursache basierend auf Dashcam-Video-Analyse und CAN-Bus-Telemetrie.
Antworte ausschließlich auf Basis der bereitgestellten Daten. Erfinde keine Fakten."""

CAUSATION_S1_PROMPT = """Analysiere den folgenden Unfall basierend auf den Wahrnehmungs- und Telemetriedaten.
Kein Zugang zu rechtlichen Referenzen verfügbar.

## Szene
{scene_description}

## Telemetrie
{telemetry_summary}

## Aufprall
{impact_details}

Bestimme:
1. Unfalltyp (rear_end, side_collision, head_on, intersection, pedestrian, single_vehicle)
2. Primäre Unfallursache
3. Beitragende Faktoren mit Schweregrad
4. Haftungsverteilung in Prozent
5. Konfidenz (0.0-1.0)

Antwort als JSON."""

CAUSATION_S2_PROMPT = """Analysiere den folgenden Unfall. Du hast Zugang zu relevantem deutschen Verkehrsrecht.

## Szene
{scene_description}

## Telemetrie
{telemetry_summary}

## Aufprall
{impact_details}

## Relevante Rechtsnormen
{legal_context}

Bestimme:
1. Unfalltyp
2. Primäre Unfallursache mit Verweis auf relevante StVO-Paragraphen
3. Beitragende Faktoren mit rechtlicher Grundlage
4. Haftungsverteilung basierend auf Rechtsprechung
5. Konfidenz (0.0-1.0)
6. Liste aller Behauptungen mit Quellenangabe

Antwort als JSON."""

CAUSATION_S3_PROMPT = """Analysiere den folgenden Unfall mit vollständiger Evidenzprüfung.
JEDE Behauptung muss durch konkrete Beweise gestützt sein.
Ungestützte Behauptungen sind NICHT erlaubt.

## Szene
{scene_description}

## Telemetrie
{telemetry_summary}

## Aufprall
{impact_details}

## Relevante Rechtsnormen
{legal_context}

## Beweismittel
{evidence_summary}

Für JEDE Behauptung angeben:
- Aussage
- Quellentyp (vision/telemetry/rag/inference)
- Quell-ID (Frame-Nr, CAN-Zeitstempel, Chunk-ID)
- Konfidenz

Bestimme:
1. Unfalltyp
2. Primäre Unfallursache (mit Beweisquelle)
3. Beitragende Faktoren (jeweils mit Beleg)
4. Haftungsverteilung (mit rechtlicher Begründung)
5. Konfidenz (0.0-1.0)
6. Vollständige Behauptungsliste mit Quellenangaben

Antwort als JSON."""

# --- Evidence Agent Prompts ---

EVIDENCE_EXTRACTION_PROMPT = """Du bist ein System zur Beweisextraktion für Unfallanalyse.

Gegeben:
- Eine Unfallbeschreibung
- Abgerufene Rechtstexte/Dokumente

Aufgabe:
- Extrahiere NUR Aussagen, die direkt zur Unfallanalyse beitragen
- Jede Aussage MUSS durch ein einzelnes Dokument belegt sein
- Füge KEIN externes Wissen hinzu

Antworte mit JSON:
[
  {{"chunk_id": "<id>", "statement": "<Aussage aus diesem Dokument>"}}
]

Leere Liste [] falls keine relevanten Beweise vorhanden."""

# --- Contradiction Agent Prompt ---

CONTRADICTION_PROMPT = """Analysiere zwei Aussagen auf logische Widersprüche.

Aussage A: {stmt_a}

Aussage B: {stmt_b}

Widersprechen sich diese Aussagen?

Antworte NUR mit JSON:
{{"contradiction": true/false, "severity": "direct"|"partial"|"tension"|"none", "explanation": "kurze Erklärung"}}"""

# --- Report Agent Prompts ---

REPORT_SYSTEM_PROMPT = """Du bist ein technischer Sachverständiger für Verkehrsunfälle.
Erstelle einen professionellen deutschen Unfallbericht (Unfallbericht).
Verwende Fachsprache und beziehe dich auf relevante Paragraphen der StVO."""

REPORT_GENERATION_PROMPT = """Erstelle einen vollständigen Unfallbericht basierend auf der folgenden Analyse:

## Unfallanalyse
{causation_analysis}

## Szenendiagramm
{scene_description}

## Telemetriedaten
{telemetry_summary}

Der Bericht soll folgende Abschnitte enthalten:
1. Unfallhergang (detaillierte Beschreibung des Ablaufs)
2. Unfallursache (technische und menschliche Faktoren)
3. Haftungsverteilung (prozentuale Zuordnung mit Begründung)
4. Schadenbeschreibung
5. Beweismittel (Liste der verwendeten Datenquellen)
6. Rechtliche Grundlagen (relevante StVO-Paragraphen und Rechtsprechung)

Antworte als strukturiertes JSON."""

# --- Validation Prompt ---

VALIDATION_PROMPT = """Du validierst eine KI-generierte Unfallanalyse.

Kontext (Beweismittel):
{context}

Analyse:
{analysis}

Ist die Analyse vollständig durch die Beweismittel gestützt?
Antworte NUR mit einer Zahl zwischen 0.0 und 1.0:
1.0 = vollständig gestützt
0.0 = halluziniert oder ungestützt"""
