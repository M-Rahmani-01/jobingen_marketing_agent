"""
agent/state.py

The single source of truth for what data flows between every agent (node)
in the LangGraph pipeline. Every node reads typed fields from this state
and writes typed fields back — never loose dicts.

Reference: Spec Section 12.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# LAYER 1 — SIGNAL
# ─────────────────────────────────────────────

class SignalItem(BaseModel):
    """One raw piece of real online activity — a Reddit post, a trends
    data point, a news headline. Never invented, always traceable."""
    source: Literal["reddit", "trends", "news", "internal"]
    text: str                  # paraphrased snippet, never a verbatim copy
    score: float                # upvotes or velocity, normalized 0-1
    url: str                    # traceability — every item must link to something real
    timestamp: Optional[str] = None


class Theme(BaseModel):
    """A cluster of related SignalItems, with a heat score, that the
    Trend Scorer will rank."""
    label: str
    evidence: list[SignalItem] = Field(default_factory=list)
    heat: float = 0.0


# ─────────────────────────────────────────────
# LAYER 2 — STRATEGY
# ─────────────────────────────────────────────

class Angle(BaseModel):
    """The differentiated take the Angle Selector picked for the chosen
    theme — must pass the Differentiation Filter (Section 9)."""
    take: str
    product_tie: Optional[str] = None       # which JobInGen feature, if any
    why_only_jobingen: str = ""              # the differentiation justification


class PlatformBrief(BaseModel):
    """One platform's creative brief from the Platform Strategist."""
    platform: Literal["linkedin", "instagram"]
    format: str              # e.g. "5-slide carousel", "single bold graphic"
    hook_style: str
    caption_length: str
    cta_type: str
    tone: str


# ─────────────────────────────────────────────
# LAYER 3 — CREATION
# ─────────────────────────────────────────────

class PostDraft(BaseModel):
    """The actual written post from the Copywriter, for one platform."""
    platform: Literal["linkedin", "instagram"]
    hook_options: list[str] = Field(default_factory=list)
    body: list[str] = Field(default_factory=list)   # slides or paragraphs
    caption: str = ""
    hashtags: list[str] = Field(default_factory=list)
    cta: str = ""
    alt_text: str = ""        # REQUIRED for every image — accessibility


# ─────────────────────────────────────────────
# LAYER 4 — ANALYSIS
# ─────────────────────────────────────────────

class Critique(BaseModel):
    """Output of one critic (platform-fit OR brand+safety)."""
    passed: bool
    score: Optional[int] = None
    feedback: str = ""                # SPECIFIC — this text gets fed back
    failed_node: Optional[str] = None  # which node should retry, if failed


class ViralityScore(BaseModel):
    """5-factor predicted-performance score (Section 11.2)."""
    total: int                    # 0-100
    hook_strength: int
    trend_alignment: int
    emotional_resonance: int
    value_density: int
    share_trigger: int
    weakest_lever: str = ""


class AnalysisReport(BaseModel):
    """The full report shipped with every approved pack (Section 11.1)."""
    trend_basis: str
    audience_tension: str
    differentiation: str
    platform_fit: str
    virality: ViralityScore
    what_to_watch: str


# ─────────────────────────────────────────────
# THE GRAPH STATE — ties everything together
# ─────────────────────────────────────────────

class RunRequest(BaseModel):
    """What kicks off a run — usually just an empty/default request,
    since the system discovers its own topic from live signal."""
    goal: Optional[str] = None
    platform: Optional[list[str]] = None   # e.g. ["linkedin", "instagram"]
    seed_topic: Optional[str] = None       # optional manual override


class MarketingState(BaseModel):
    """
    The ONE object that flows through every node in the LangGraph.
    Each node reads the fields it needs and writes the fields it produces.
    Nothing is shared between nodes except through this object.
    """
    request: RunRequest = Field(default_factory=RunRequest)

    # Layer 1 outputs
    raw_signal: list[SignalItem] = Field(default_factory=list)
    themes: list[Theme] = Field(default_factory=list)
    chosen_theme: Optional[Theme] = None     # None = "no strong theme" (valid!)

    # Layer 2 outputs
    tension: Optional[str] = None
    angle: Optional[Angle] = None
    briefs: dict[str, PlatformBrief] = Field(default_factory=dict)

    # Layer 3 outputs
    drafts: dict[str, PostDraft] = Field(default_factory=dict)
    images: dict[str, list[str]] = Field(default_factory=dict)  # file paths

    # Layer 4 outputs
    critiques: dict[str, Critique] = Field(default_factory=dict)
    virality: dict[str, ViralityScore] = Field(default_factory=dict)
    retry_counts: dict[str, int] = Field(default_factory=dict)
    report: dict[str, AnalysisReport] = Field(default_factory=dict)

    # Control flow
    approved: Optional[bool] = None
    errors: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}