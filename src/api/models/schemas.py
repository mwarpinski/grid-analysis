"""
Pydantic models for API request/response validation.

Shared across all routers so every endpoint speaks the same schema.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

class ConstraintSnapshot(BaseModel):
    """One binding constraint record at a specific interval."""
    interval: datetime
    constraint_name: str
    canonical_id: Optional[str] = None      # constraint_key: monitored element + contingency
    element_key: Optional[str] = None       # the monitored element alone
    shadow_price: float
    limit_mw: Optional[float] = None
    monitored_mw: Optional[float] = None
    pct_of_limit: Optional[float] = None
    state: Optional[str] = None
    contingency_name: Optional[str] = None
    volatility_score: Optional[float] = None


class LiveConstraintsResponse(BaseModel):
    as_of: datetime
    interval: Optional[datetime]
    n_binding: int
    constraints: List[ConstraintSnapshot]


# ---------------------------------------------------------------------------
# Shift factors
# ---------------------------------------------------------------------------

class ShiftFactorRecord(BaseModel):
    node: str
    constraint: str
    shift_factor: float
    shadow_price: Optional[float] = None
    sf_exposure: Optional[float] = None   # GSF × SP — the key trader metric


class ShiftFactorTableResponse(BaseModel):
    constraint: str
    shadow_price: Optional[float]
    n_nodes: int
    records: List[ShiftFactorRecord]


class SingleSFResponse(BaseModel):
    node: str
    constraint: str
    shift_factor: Optional[float]
    shadow_price: Optional[float] = None
    sf_exposure: Optional[float] = None
    solved: bool


# ---------------------------------------------------------------------------
# Flow impact (bilateral what-if)
# ---------------------------------------------------------------------------

class FlowImpactRequest(BaseModel):
    source_node: str = Field(..., description="Injection node")
    sink_node: str = Field(..., description="Withdrawal node")
    mw: float = Field(..., gt=0, description="Bilateral MW injection (must be positive)")
    constraint: str = Field(..., description="Constraint name to evaluate")
    iso: Optional[str] = Field("spp", description="ISO namespace: spp or miso")


class FlowImpactResponse(BaseModel):
    source_node: str
    sink_node: str
    mw: float
    constraint: str
    delta_flow_mw: Optional[float]
    direction: Optional[str]       # "exacerbates" | "relieves" | "neutral"
    sf_source: Optional[float]
    sf_sink: Optional[float]
    solved: bool
    message: str


# ---------------------------------------------------------------------------
# MCC (pre-clearing LMP congestion component)
# ---------------------------------------------------------------------------

class MCCContribution(BaseModel):
    constraint: str
    shadow_price: float
    shift_factor: Optional[float]
    contribution: float            # SF × SP (0 when no shift factor is available)
    source: Optional[str] = None   # "fitted", "sibling_prior" (borrowed from the same element), or None


class MCCResponse(BaseModel):
    node: str
    mcc: float                     # Σ(SF × SP)
    n_constraints_used: int
    # Estimate that also uses provisional shift factors borrowed from sibling constraints (same
    # physical element, another contingency). `mcc` itself uses fitted shift factors only.
    mcc_with_priors: Optional[float] = None
    shadow_price_coverage_with_priors: Optional[float] = None
    # Share of the current Σ|shadow price| carried by constraints that HAVE shift
    # factors. The estimate silently leaves out the rest, so a value well below 1.0
    # means `mcc` is missing most of the congestion (e.g. a newly binding constraint).
    shadow_price_coverage: Optional[float] = None
    shadow_prices_used: List[MCCContribution]


# ---------------------------------------------------------------------------
# Portfolio positions
# ---------------------------------------------------------------------------

class Position(BaseModel):
    asset_name: str
    asset_type: str = Field(
        ...,
        description="'gen', 'load', 'virtual_buy', 'virtual_sell', 'ftr', 'other'"
    )
    node: str
    mw_signed: float = Field(
        ...,
        description="Positive = injection/generation, Negative = withdrawal/load"
    )

    @field_validator("asset_type")
    @classmethod
    def valid_asset_type(cls, v: str) -> str:
        allowed = {"gen", "load", "virtual_buy", "virtual_sell", "ftr", "other"}
        if v.lower() not in allowed:
            raise ValueError(f"asset_type must be one of {allowed}")
        return v.lower()


class PositionRiskRecord(BaseModel):
    asset_name: str
    asset_type: str
    node: str
    mw_signed: float
    constraint: str
    gsf: Optional[float]
    net_mw_flow: Optional[float]      # GSF × mw_signed
    shadow_price: float
    pnl_risk_rate: Optional[float]    # net_mw_flow × shadow_price ($/hr)
    direction: Optional[str]          # "exacerbates" | "relieves" | "neutral"


class PortfolioRiskResponse(BaseModel):
    constraint: str
    shadow_price: float
    total_net_mw_flow: float
    total_pnl_risk_rate: float        # $/hr at current shadow price
    alert_message: str
    positions: List[PositionRiskRecord]


class PositionsUploadResponse(BaseModel):
    accepted: int
    positions: List[Position]


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

class ReplayInterval(BaseModel):
    interval: datetime
    n_binding_constraints: int


class ReplaySnapshotResponse(BaseModel):
    interval: datetime
    constraints: List[ConstraintSnapshot]
    shift_factors: List[ShiftFactorRecord]


class ReplayPnLPoint(BaseModel):
    interval: datetime
    pnl_rate_per_hr: float
    n_binding_constraints: int
    dominant_constraint: Optional[str]


class ReplayPnLResponse(BaseModel):
    start: datetime
    end: datetime
    n_intervals: int
    total_pnl: float
    series: List[ReplayPnLPoint]
