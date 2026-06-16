"""
Configurable RDKit filters for PubChem CSV rows (PUBCHEM_COMPOUND_CID, SMILES).

Used by filter_molecules_configurable.py for JSON / CLI configured filtering
driven by the pubchem-mol-filter agent skill.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set

from rdkit import Chem
from rdkit.Chem import Descriptors

DEFAULT_ALLOWED_ELEMENTS = ["H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
HETEROATOM_NOSP = {"N", "O", "S", "P"}
HETEROATOM_VALENCE_ELECTRONS = {"N": 5, "O": 6, "S": 6, "P": 5}

BASE_REJECTION_KEYS = ["null_mol", "no_smiles"]


@dataclass
class FilterConfig:
    """Per-filter enable flags and thresholds. Disabled filters are skipped."""

    # No valence / sanitization errors
    require_sanitization: bool = True
    # Exactly one connected component
    require_single_component: bool = True
    # Allowed element symbols (empty list disables this check)
    allowed_elements: List[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_ELEMENTS))
    # At least one of these elements present (composition); empty disables
    require_any_elements: List[str] = field(default_factory=lambda: list(HETEROATOM_NOSP))
    # All of these elements must be present; empty disables
    require_all_elements: List[str] = field(default_factory=list)
    # Reject if any of these elements present; empty disables
    forbidden_elements: List[str] = field(default_factory=list)
    # Heavy atom count (non-H) upper bound; None disables
    max_heavy_atoms: Optional[int] = 30
    # Reject molecules with radical electrons on any atom
    reject_radicals: bool = True
    # Largest ring size upper bound; None disables
    max_ring_size: Optional[int] = 6
    # Molecular weight upper bound (Da); None disables
    max_mol_weight: Optional[float] = 500.0
    # Reject zwitterions (+ and - formal charge on different atoms)
    reject_zwitterion: bool = True
    # Reject isotope-labeled atoms
    reject_isotopes: bool = True
    # At least one N/O/S/P with a lone pair; disabled if require_any_elements empty
    require_heteroatom_lone_pair: bool = True

    def allowed_element_set(self) -> Set[str]:
        return set(self.allowed_elements)

    def active_filter_names(self) -> List[str]:
        names = []
        if self.require_sanitization:
            names.append("sanitization")
        if self.require_single_component:
            names.append("single_component")
        if self.allowed_elements:
            names.append("allowed_elements")
        if self.require_any_elements:
            names.append("require_any_elements")
        if self.require_all_elements:
            names.append("require_all_elements")
        if self.forbidden_elements:
            names.append("forbidden_elements")
        if self.max_heavy_atoms is not None:
            names.append("max_heavy_atoms")
        if self.reject_radicals:
            names.append("no_radicals")
        if self.max_ring_size is not None:
            names.append("max_ring_size")
        if self.max_mol_weight is not None:
            names.append("max_mol_weight")
        if self.reject_zwitterion:
            names.append("no_zwitterion")
        if self.reject_isotopes:
            names.append("no_isotope")
        if self.require_heteroatom_lone_pair and self.require_any_elements:
            names.append("heteroatom_lone_pair")
        return names

    def all_rejection_keys(self) -> List[str]:
        return self.active_filter_names() + BASE_REJECTION_KEYS


def default_filter_config() -> FilterConfig:
    """Reasonable default criteria; copy/edit per the user's request."""
    return FilterConfig()


def config_from_dict(data: Dict) -> FilterConfig:
    """Build FilterConfig from a JSON-compatible dict (unknown keys ignored)."""
    fields = {f.name for f in FilterConfig.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in data.items() if k in fields}
    return FilterConfig(**kwargs)


def load_config_json(path: str) -> FilterConfig:
    with open(path, "r", encoding="utf-8") as f:
        return config_from_dict(json.load(f))


def save_config_json(config: FilterConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
        f.write("\n")


@dataclass
class IOConfig:
    """Input/output and execution settings for a filter run.

    Everything an agent platform needs to execute the skill lives here, so a
    single JSON config fully specifies a run with no extra CLI arguments.
    """

    # "single" (one CSV) or "shards" (a directory of *.csv)
    mode: str = "single"
    # mode == "single"
    input: Optional[str] = None
    output: Optional[str] = None
    # mode == "shards"
    input_dir: Optional[str] = None
    output_dir: Optional[str] = None
    # Parallel worker processes
    workers: int = 64
    # Where to archive the resolved config actually used for the run
    save_config_used: Optional[str] = "filter_config_used.json"

    _PLACEHOLDER_MARKERS = ("/ABSOLUTE/PATH", "PLACEHOLDER", "TODO", "CHANGEME", "YOUR_PATH")

    def _check_path(self, path: Optional[str], field: str) -> None:
        if not path:
            return
        lower = path.lower()
        for marker in self._PLACEHOLDER_MARKERS:
            if marker.lower() in lower:
                raise ValueError(
                    f"io.{field} looks like an unconfirmed placeholder ({path!r}). "
                    "Set a real absolute path after user confirmation."
                )

    def validate(self) -> None:
        if self.mode not in ("single", "shards"):
            raise ValueError(f"io.mode must be 'single' or 'shards', got {self.mode!r}")
        if self.mode == "single":
            if not self.input:
                raise ValueError("io.input is required when io.mode == 'single'")
            if not self.output:
                raise ValueError("io.output is required when io.mode == 'single'")
        else:
            if not self.input_dir:
                raise ValueError("io.input_dir is required when io.mode == 'shards'")
            if not self.output_dir:
                raise ValueError("io.output_dir is required when io.mode == 'shards'")
        if self.workers < 1:
            raise ValueError("io.workers must be >= 1")
        if self.mode == "single":
            self._check_path(self.input, "input")
            self._check_path(self.output, "output")
        else:
            self._check_path(self.input_dir, "input_dir")
            self._check_path(self.output_dir, "output_dir")
        self._check_path(self.save_config_used, "save_config_used")


@dataclass
class RunConfig:
    """Full skill run specification: I/O settings + filter criteria."""

    io: IOConfig = field(default_factory=IOConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    # Must be true before execution — set only after user confirms all fields
    confirmed: bool = False


def io_config_from_dict(data: Dict) -> IOConfig:
    fields = {f.name for f in IOConfig.__dataclass_fields__.values()}
    return IOConfig(**{k: v for k, v in data.items() if k in fields})


def default_run_config() -> RunConfig:
    return RunConfig(io=IOConfig(), filters=FilterConfig())


def run_config_from_dict(data: Dict) -> RunConfig:
    """Build a RunConfig from a dict.

    Accepts the wrapped form {"io": {...}, "filters": {...}} or a flat/legacy
    FilterConfig dict (in which case I/O falls back to defaults).
    """
    if "filters" in data or "io" in data:
        io = io_config_from_dict(data.get("io", {}))
        filters = config_from_dict(data.get("filters", {}))
        confirmed = bool(data.get("confirmed", False))
    else:
        io = IOConfig()
        filters = config_from_dict(data)
        confirmed = bool(data.get("confirmed", False))
    return RunConfig(io=io, filters=filters, confirmed=confirmed)


def load_run_config_json(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        return run_config_from_dict(json.load(f))


def run_config_to_dict(run: RunConfig) -> Dict:
    return {"confirmed": run.confirmed, "io": asdict(run.io), "filters": asdict(run.filters)}


def save_run_config_json(run: RunConfig, path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run_config_to_dict(run), f, indent=2)
        f.write("\n")


def atom_has_lone_pair(atom: Chem.Atom) -> bool:
    ve = HETEROATOM_VALENCE_ELECTRONS.get(atom.GetSymbol())
    if ve is None:
        return False
    return (ve - atom.GetTotalValence() - atom.GetFormalCharge()) >= 2


def is_zwitterion(mol: Chem.Mol) -> bool:
    has_pos = has_neg = False
    for atom in mol.GetAtoms():
        fc = atom.GetFormalCharge()
        if fc > 0:
            has_pos = True
        elif fc < 0:
            has_neg = True
        if has_pos and has_neg:
            return True
    return False


def check_filters(mol: Optional[Chem.Mol], config: FilterConfig) -> Optional[str]:
    """
    Apply enabled filters. Return None if passed, else rejection reason string.
    """
    if mol is None:
        return "null_mol"

    if config.require_sanitization:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return "sanitization"

    if config.require_single_component:
        if len(Chem.GetMolFrags(mol, asMols=True)) != 1:
            return "single_component"

    atoms = list(mol.GetAtoms())
    symbols = [a.GetSymbol() for a in atoms]
    symbol_set = set(symbols)

    if config.allowed_elements:
        allowed = config.allowed_element_set()
        for sym in symbols:
            if sym not in allowed:
                return "allowed_elements"

    if config.require_any_elements:
        if not symbol_set.intersection(config.require_any_elements):
            return "require_any_elements"

    if config.require_all_elements:
        missing = set(config.require_all_elements) - symbol_set
        if missing:
            return "require_all_elements"

    if config.forbidden_elements:
        if symbol_set.intersection(config.forbidden_elements):
            return "forbidden_elements"

    if config.max_heavy_atoms is not None:
        heavy = sum(1 for a in atoms if a.GetAtomicNum() > 1)
        if heavy > config.max_heavy_atoms:
            return "max_heavy_atoms"

    if config.reject_radicals:
        if any(a.GetNumRadicalElectrons() != 0 for a in atoms):
            return "no_radicals"

    if config.max_ring_size is not None:
        for ring in mol.GetRingInfo().AtomRings():
            if len(ring) > config.max_ring_size:
                return "max_ring_size"

    if config.max_mol_weight is not None:
        if Descriptors.MolWt(mol) > config.max_mol_weight:
            return "max_mol_weight"

    if config.reject_zwitterion:
        if is_zwitterion(mol):
            return "no_zwitterion"

    if config.reject_isotopes:
        if any(a.GetIsotope() != 0 for a in atoms):
            return "no_isotope"

    if config.require_heteroatom_lone_pair and config.require_any_elements:
        hetero_syms = set(config.require_any_elements) & HETEROATOM_VALENCE_ELECTRONS.keys()
        heteroatoms = [a for a in atoms if a.GetSymbol() in hetero_syms]
        if heteroatoms and not any(atom_has_lone_pair(a) for a in heteroatoms):
            return "heteroatom_lone_pair"

    return None
