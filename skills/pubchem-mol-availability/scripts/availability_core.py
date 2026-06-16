"""
Core data model and offline (RDKit / text) logic for the pubchem-mol-availability
skill.

Responsibilities (all pure / offline — no network):
- RunConfig / IOConfig / LookupConfig / LLMConfig / FilterConfig dataclasses
  with a `confirmed` execution gate and placeholder-path validation.
- Halide-salt classification of a SMILES with RDKit.
- Physical-form normalization from free text (powder / solid / liquid / gas / unknown).
- Melting-point parsing (°C / °F) from free text.
- The deterministic form cascade resolver (PubChem text -> MP heuristic -> LLM).

Network access (PubChem PUG REST/View, LLM API) lives in pubchem_client.py.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

# Counterions treated as "halide" for halide-salt detection.
HALIDE_SYMBOLS = {"F", "Cl", "Br", "I"}

# Monoatomic metal counterions commonly seen in salts (alkali/alkaline/transition).
METAL_SYMBOLS = {
    "Li", "Na", "K", "Rb", "Cs",
    "Be", "Mg", "Ca", "Sr", "Ba",
    "Al", "Zn", "Fe", "Cu", "Mn", "Ag", "Ni", "Co", "Pt", "Pd", "Au",
}

# Free-text physical-form vocabulary. Order of checks matters (see normalize_form).
_POWDER_KEYWORDS = ("powder",)
_SOLID_KEYWORDS = (
    "crystal",
    "crystalline",
    "needle",
    "flake",
    "granule",
    "prism",
    "plate",
    "leaflet",
    "pellet",
    "lump",
    "solid",
    "wax",
)
_LIQUID_KEYWORDS = ("liquid", "oil", "oily", "syrup")
_GAS_KEYWORDS = ("gas", "gaseous")

VALID_FORMS = ("powder", "solid", "liquid", "gas", "unknown")
_PLACEHOLDER_MARKERS = ("/REPLACE", "/ABSOLUTE/PATH", "PLACEHOLDER", "TODO", "CHANGEME", "YOUR_PATH")


# --------------------------------------------------------------------------- #
# Configuration dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class IOConfig:
    input: Optional[str] = None
    output: Optional[str] = None
    dropped_output: Optional[str] = None
    cache_dir: Optional[str] = None
    workers: int = 64
    # Optional explicit input column names. When null, the reader auto-detects
    # from PUBCHEM_COMPOUND_CID/cid/CID and SMILES/smiles.
    cid_column: Optional[str] = None
    smiles_column: Optional[str] = None

    def _check_path(self, path: Optional[str], name: str) -> None:
        if not path:
            return
        lower = path.lower()
        for marker in _PLACEHOLDER_MARKERS:
            if marker.lower() in lower:
                raise ValueError(
                    f"io.{name} looks like an unconfirmed placeholder ({path!r}). "
                    "Set a real absolute path after user confirmation."
                )

    def validate(self) -> None:
        if not self.input:
            raise ValueError("io.input is required (CSV with cid, smiles columns)")
        if not self.output:
            raise ValueError("io.output is required (final annotated CSV path)")
        if self.workers < 1:
            raise ValueError("io.workers must be >= 1")
        self._check_path(self.input, "input")
        self._check_path(self.output, "output")
        self._check_path(self.dropped_output, "dropped_output")
        self._check_path(self.cache_dir, "cache_dir")


@dataclass
class LookupConfig:
    request_rate_per_sec: float = 5.0
    max_retries: int = 4
    timeout_sec: int = 30
    # Only count F/Cl/Br/I counterions as a "salt form" (per skill scope).
    halide_only: bool = True
    # Also enrich + report the neutral input compound, not just its salts.
    report_neutral: bool = True
    # "per_salt": one output row per halide-salt CID; "rollup": one row per input.
    row_granularity: str = "per_salt"
    # Melting point (deg C) above which a compound is treated as solid.
    mp_solid_threshold_c: float = 25.0

    def validate(self) -> None:
        if self.request_rate_per_sec <= 0:
            raise ValueError("lookup.request_rate_per_sec must be > 0")
        if self.row_granularity not in ("per_salt", "rollup"):
            raise ValueError("lookup.row_granularity must be 'per_salt' or 'rollup'")


@dataclass
class LLMConfig:
    enabled: bool = True
    model: str = "gpt-5.5"
    temperature: float = 0.0
    confidence_threshold: float = 0.7
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    max_retries: int = 3

    def validate(self) -> None:
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("llm.confidence_threshold must be in [0, 1]")


@dataclass
class FilterConfig:
    # Drop rows whose decision form is confirmed 'liquid'. 'unknown' is kept.
    drop_liquids: bool = True


@dataclass
class RunConfig:
    io: IOConfig = field(default_factory=IOConfig)
    lookup: LookupConfig = field(default_factory=LookupConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    confirmed: bool = False

    def validate(self) -> None:
        self.io.validate()
        self.lookup.validate()
        self.llm.validate()


def _subset(cls, data: Dict):
    fields = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in data.items() if k in fields})


def default_run_config() -> RunConfig:
    return RunConfig()


def run_config_from_dict(data: Dict) -> RunConfig:
    return RunConfig(
        io=_subset(IOConfig, data.get("io", {})),
        lookup=_subset(LookupConfig, data.get("lookup", {})),
        llm=_subset(LLMConfig, data.get("llm", {})),
        filter=_subset(FilterConfig, data.get("filter", {})),
        confirmed=bool(data.get("confirmed", False)),
    )


def run_config_to_dict(run: RunConfig) -> Dict:
    return {
        "confirmed": run.confirmed,
        "io": asdict(run.io),
        "lookup": asdict(run.lookup),
        "llm": asdict(run.llm),
        "filter": asdict(run.filter),
    }


def load_run_config_json(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        return run_config_from_dict(json.load(f))


def save_run_config_json(run: RunConfig, path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run_config_to_dict(run), f, indent=2)
        f.write("\n")


# --------------------------------------------------------------------------- #
# Halide-salt classification (RDKit, offline)
# --------------------------------------------------------------------------- #
@dataclass
class SaltInfo:
    is_salt: bool
    n_components: int
    counterions: List[str]
    is_halide_salt: bool
    halide_counterions: List[str]


def _fragment_counterion_symbol(frag: Chem.Mol) -> Optional[str]:
    """Return the element symbol if a fragment is a simple monoatomic counterion.

    Catches both charged and neutral monoatomic representations of halide
    counterions (e.g. ``[Cl-]`` and the free-acid form ``Cl`` in ``CCN.Cl``) and
    simple metal cations (Na+, K+, ...). Returns None for the organic component,
    polyatomic ions, and neutral monoatomic non-counterions (e.g. water oxygen in
    a hydrate, which must not be mistaken for a salt).
    """
    heavy = [a for a in frag.GetAtoms() if a.GetAtomicNum() > 1]
    if len(heavy) != 1:
        return None
    atom = heavy[0]
    symbol = atom.GetSymbol()
    if atom.GetFormalCharge() != 0:
        return symbol
    if symbol in HALIDE_SYMBOLS or symbol in METAL_SYMBOLS:
        return symbol
    return None


def classify_salt(smiles: Optional[str]) -> SaltInfo:
    """Classify a SMILES as salt / halide-salt using fragment analysis."""
    empty = SaltInfo(False, 0, [], False, [])
    if not smiles:
        return empty
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return empty
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    except Exception:
        return empty
    n = len(frags)
    if n < 2:
        return SaltInfo(False, n, [], False, [])

    counterions: List[str] = []
    for frag in frags:
        sym = _fragment_counterion_symbol(frag)
        if sym is not None:
            counterions.append(sym)

    is_salt = len(counterions) >= 1
    halide_counterions = [s for s in counterions if s in HALIDE_SYMBOLS]
    return SaltInfo(
        is_salt=is_salt,
        n_components=n,
        counterions=counterions,
        is_halide_salt=len(halide_counterions) >= 1,
        halide_counterions=halide_counterions,
    )


# --------------------------------------------------------------------------- #
# Physical-form normalization + melting-point parsing (text, offline)
# --------------------------------------------------------------------------- #
def normalize_form(text: Optional[str]) -> str:
    """Map free-text physical description to powder/solid/liquid/gas/unknown.

    'powder' is reported distinctly; it is a (kept) solid morphology. Checks are
    ordered so a specific morphology word wins over a generic state word.
    """
    if not text:
        return "unknown"
    t = text.lower()
    if any(k in t for k in _POWDER_KEYWORDS):
        return "powder"
    if any(k in t for k in _SOLID_KEYWORDS):
        return "solid"
    if any(k in t for k in _LIQUID_KEYWORDS):
        return "liquid"
    if any(k in t for k in _GAS_KEYWORDS):
        return "gas"
    return "unknown"


_MP_PATTERN = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:(?:-|to|–)\s*-?\d+(?:\.\d+)?)?\s*°?\s*([CF])\b",
    re.IGNORECASE,
)


def parse_melting_point_c(text: Optional[str]) -> Optional[float]:
    """Extract a melting point in degrees Celsius from free text.

    Takes the first numeric value (lower bound of a range) and converts °F to °C.
    Returns None when no temperature is found.
    """
    if not text:
        return None
    m = _MP_PATTERN.search(text)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    unit = m.group(2).upper()
    if unit == "F":
        value = (value - 32.0) * 5.0 / 9.0
    return value


# --------------------------------------------------------------------------- #
# Form cascade resolver
# --------------------------------------------------------------------------- #
@dataclass
class FormVerdict:
    form: str            # powder / solid / liquid / gas / unknown
    source: str          # pubchem / mp_heuristic / llm / none
    confidence: Optional[float]


def resolve_form(
    description_text: Optional[str],
    melting_point_c: Optional[float],
    mp_solid_threshold_c: float,
    llm_form: Optional[str] = None,
    llm_confidence: Optional[float] = None,
    llm_confidence_threshold: float = 0.7,
) -> FormVerdict:
    """Run the deterministic cascade: PubChem text -> MP heuristic -> LLM.

    `llm_form`/`llm_confidence` are supplied by the caller only when the first two
    steps yield 'unknown' (so the LLM is queried lazily). Confidence below the
    threshold is treated as 'unknown' rather than forcing a guess.
    """
    form = normalize_form(description_text)
    if form != "unknown":
        return FormVerdict(form=form, source="pubchem", confidence=None)

    if melting_point_c is not None:
        form = "solid" if melting_point_c > mp_solid_threshold_c else "liquid"
        return FormVerdict(form=form, source="mp_heuristic", confidence=None)

    if llm_form in ("solid", "liquid", "powder"):
        if llm_confidence is None or llm_confidence >= llm_confidence_threshold:
            return FormVerdict(form=llm_form, source="llm", confidence=llm_confidence)

    return FormVerdict(form="unknown", source="none", confidence=None)


def needs_llm(description_text: Optional[str], melting_point_c: Optional[float]) -> bool:
    """True when both PubChem text and MP are inconclusive (LLM fallback needed)."""
    return normalize_form(description_text) == "unknown" and melting_point_c is None
