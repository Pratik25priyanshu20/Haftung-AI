"""Causation analysis output types for Haftung_AI."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """A single factual claim with evidence attribution."""

    statement: str = Field(description="The factual claim text")
    source_type: str = Field(description="Source: 'vision', 'telemetry', 'rag', 'inference'")
    source_id: str | None = Field(default=None, description="Chunk ID, frame ID, or CAN timestamp")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this claim")
    supported: bool | None = Field(default=None, description="Whether claim is supported by evidence")


class ContributingFactor(BaseModel):
    """A factor that contributed to the accident."""

    factor: str = Field(description="Description of the contributing factor")
    category: str = Field(description="Category: 'speed', 'distance', 'visibility', 'road_condition', 'driver_behavior', 'vehicle_defect', 'other'")
    severity: str = Field(description="Severity: 'primary', 'secondary', 'minor'")
    legal_reference: str | None = Field(default=None, description="StVO paragraph or case law reference")


class ResponsibilityAssignment(BaseModel):
    """Responsibility distribution between parties."""

    party: str = Field(description="Party identifier: 'ego', 'other_1', 'other_2', etc.")
    percentage: float = Field(ge=0.0, le=100.0, description="Responsibility percentage")
    rationale: str = Field(description="Legal/factual basis for this assignment")


class CausationOutput(BaseModel):
    """Complete causation analysis output."""

    accident_type: str = Field(description="Classification: 'rear_end', 'side_collision', 'head_on', 'intersection', 'pedestrian', 'single_vehicle'")
    primary_cause: str = Field(description="Primary cause of the accident")
    contributing_factors: list[ContributingFactor] = Field(default_factory=list)
    responsibility: list[ResponsibilityAssignment] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    legal_references: list[str] = Field(default_factory=list, description="Relevant StVO/case law")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall analysis confidence")
    variant: str = Field(description="System variant: 'S1', 'S2', 'S3'")
    reasoning: str = Field(default="", description="Chain-of-thought reasoning")
