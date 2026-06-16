"""
Network clients for the pubchem-mol-availability skill.

- JsonCache: simple persistent dict cache (idempotent re-runs / resume).
- PubChemClient: throttled PUG REST + PUG View lookups
  (same-parent CIDs, SMILES, physical description, melting point, vendor count).
- LLMClient: powder-vs-liquid classification fallback via an OpenAI-style API.

All HTTP errors are caught and surfaced as empty / None results so a batch run
degrades gracefully (fields become 'unknown') instead of crashing.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest"


# --------------------------------------------------------------------------- #
# Persistent JSON cache
# --------------------------------------------------------------------------- #
class JsonCache:
    """Thread-safe dict persisted to a JSON file; flushed periodically."""

    def __init__(self, path: str, flush_every: int = 25):
        self.path = path
        self.flush_every = flush_every
        self._lock = threading.Lock()
        self._dirty = 0
        self._data: Dict[str, Any] = {}
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def get(self, key: str, default=None):
        with self._lock:
            return self._data.get(key, default)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._dirty += 1
            if self._dirty >= self.flush_every:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self.path:
            return
        out_dir = os.path.dirname(self.path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f)
        os.replace(tmp, self.path)
        self._dirty = 0


# --------------------------------------------------------------------------- #
# Recursive helpers for PUG View JSON
# --------------------------------------------------------------------------- #
def _collect_strings(node: Any, out: List[str]) -> None:
    """Collect all StringWithMarkup 'String' values found anywhere in a record."""
    if isinstance(node, dict):
        if "StringWithMarkup" in node and isinstance(node["StringWithMarkup"], list):
            for swm in node["StringWithMarkup"]:
                s = swm.get("String") if isinstance(swm, dict) else None
                if isinstance(s, str):
                    out.append(s)
        for v in node.values():
            _collect_strings(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect_strings(v, out)


def _pugview_section_strings(record: Dict, heading: str) -> List[str]:
    strings: List[str] = []
    _collect_strings(record, strings)
    return strings


# --------------------------------------------------------------------------- #
# PubChem client
# --------------------------------------------------------------------------- #
class PubChemClient:
    def __init__(
        self,
        cache: JsonCache,
        request_rate_per_sec: float = 5.0,
        max_retries: int = 4,
        timeout_sec: int = 30,
    ):
        self.cache = cache
        self.min_interval = 1.0 / max(request_rate_per_sec, 0.1)
        self.max_retries = max_retries
        self.timeout = timeout_sec
        self._last_request = 0.0
        self._rate_lock = threading.Lock()
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "pubchem-mol-availability/1.0"})

    def _throttle(self) -> None:
        with self._rate_lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    def _get_json(self, url: str) -> Optional[Dict]:
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = self._session.get(url, timeout=self.timeout)
            except requests.RequestException:
                time.sleep(min(2 ** attempt, 10))
                continue
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    return None
            # 404 = no data for this CID/section; not an error worth retrying.
            if resp.status_code == 404:
                return None
            # 429 / 503 -> back off and retry.
            if resp.status_code in (429, 503):
                time.sleep(min(2 ** attempt, 10))
                continue
            return None
        return None

    # ---- PUG REST -------------------------------------------------------- #
    def same_parent_cids(self, cid: int) -> List[int]:
        key = f"same_parent:{cid}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        url = (
            f"{PUBCHEM_BASE}/pug/compound/cid/{cid}/cids/JSON"
            "?cids_type=same_parent_connectivity"
        )
        data = self._get_json(url)
        related: List[int] = []
        if data:
            try:
                related = list(data["IdentifierList"]["CID"])
            except (KeyError, TypeError):
                related = []
        related = [c for c in related if c != cid]
        self.cache.set(key, related)
        return related

    def smiles_for_cids(self, cids: List[int]) -> Dict[int, str]:
        """Batch-fetch isomeric (fallback canonical) SMILES for CIDs."""
        result: Dict[int, str] = {}
        missing: List[int] = []
        for c in cids:
            cached = self.cache.get(f"smiles:{c}")
            if cached is not None:
                result[c] = cached
            else:
                missing.append(c)
        for chunk in _chunks(missing, 100):
            joined = ",".join(str(c) for c in chunk)
            url = (
                f"{PUBCHEM_BASE}/pug/compound/cid/{joined}"
                "/property/IsomericSMILES,CanonicalSMILES/JSON"
            )
            data = self._get_json(url)
            props = []
            if data:
                try:
                    props = data["PropertyTable"]["Properties"]
                except (KeyError, TypeError):
                    props = []
            for p in props:
                c = p.get("CID")
                smi = p.get("IsomericSMILES") or p.get("CanonicalSMILES") or ""
                if c is not None:
                    result[int(c)] = smi
                    self.cache.set(f"smiles:{c}", smi)
            # Cache misses as empty so we don't re-query forever.
            for c in chunk:
                if c not in result:
                    result[c] = ""
                    self.cache.set(f"smiles:{c}", "")
        return result

    # ---- PUG View -------------------------------------------------------- #
    def _pugview(self, cid: int, heading: str) -> Optional[Dict]:
        heading_q = requests.utils.quote(heading)
        url = (
            f"{PUBCHEM_BASE}/pug_view/data/compound/{cid}/JSON?heading={heading_q}"
        )
        return self._get_json(url)

    def physical_description(self, cid: int) -> str:
        key = f"phys_desc:{cid}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        text_parts: List[str] = []
        for heading in ("Physical Description", "Color/Form"):
            rec = self._pugview(cid, heading)
            if rec:
                text_parts.extend(_pugview_section_strings(rec, heading))
        text = " | ".join(dict.fromkeys(text_parts))  # dedupe, keep order
        self.cache.set(key, text)
        return text

    def melting_point_text(self, cid: int) -> str:
        key = f"mp_text:{cid}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        rec = self._pugview(cid, "Melting Point")
        parts: List[str] = []
        if rec:
            parts = _pugview_section_strings(rec, "Melting Point")
        text = " | ".join(dict.fromkeys(parts))
        self.cache.set(key, text)
        return text

    def vendor_info(self, cid: int) -> Tuple[int, List[str]]:
        """Return (vendor_count, example_vendor_names) from the Chemical Vendors category."""
        key = f"vendors:{cid}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached.get("count", 0), cached.get("examples", [])
        url = f"{PUBCHEM_BASE}/pug_view/categories/compound/{cid}/JSON"
        data = self._get_json(url)
        count = 0
        examples: List[str] = []
        if data:
            categories = []
            try:
                categories = data["SourceCategories"]["Categories"]
            except (KeyError, TypeError):
                categories = []
            for cat in categories:
                if str(cat.get("Category", "")).lower().startswith("chemical vendor"):
                    sources = cat.get("Sources", []) or []
                    count = len(sources)
                    for s in sources[:5]:
                        name = s.get("SourceName")
                        if name:
                            examples.append(name)
                    break
        self.cache.set(key, {"count": count, "examples": examples})
        return count, examples


# --------------------------------------------------------------------------- #
# LLM client (powder/liquid fallback classifier)
# --------------------------------------------------------------------------- #
_LLM_SYSTEM = (
    "You are a chemistry expert. Given a molecule, decide whether the pure "
    "compound at room temperature (25 C, 1 atm) is typically a SOLID "
    "(crystalline/powder) or a LIQUID. Respond ONLY with compact JSON: "
    '{"form": "solid"|"liquid", "confidence": <0-1 float>, "reasoning": "<short>"}.'
)


class LLMClient:
    """OpenAI-style chat classifier with on-disk caching. Disabled cleanly if
    the SDK or API key is unavailable (results fall through to 'unknown')."""

    def __init__(self, config, cache: JsonCache):
        self.cfg = config
        self.cache = cache
        self._client = None
        self._unavailable_reason: Optional[str] = None
        if not config.enabled:
            self._unavailable_reason = "llm.enabled is false"
            return
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            self._unavailable_reason = f"env var {config.api_key_env} not set"
            return
        try:
            from openai import OpenAI
        except Exception:
            self._unavailable_reason = "openai package not installed"
            return
        kwargs = {"api_key": api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        try:
            self._client = OpenAI(**kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            self._unavailable_reason = f"OpenAI client init failed: {exc}"

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def unavailable_reason(self) -> Optional[str]:
        return self._unavailable_reason

    def classify(self, cid: int, smiles: str) -> Tuple[Optional[str], Optional[float]]:
        key = f"llm:{self.cfg.model}:{cid}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached.get("form"), cached.get("confidence")
        if not self.available:
            return None, None

        user = f"Molecule SMILES: {smiles}\nPubChem CID: {cid}\nClassify solid vs liquid."
        form: Optional[str] = None
        confidence: Optional[float] = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.cfg.model,
                    temperature=self.cfg.temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": _LLM_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                )
                content = resp.choices[0].message.content
                parsed = json.loads(content)
                raw_form = str(parsed.get("form", "")).strip().lower()
                if raw_form in ("solid", "liquid"):
                    form = raw_form
                confidence = parsed.get("confidence")
                if isinstance(confidence, (int, float)):
                    confidence = float(confidence)
                else:
                    confidence = None
                break
            except Exception:
                time.sleep(min(2 ** attempt, 8))
                continue

        self.cache.set(key, {"form": form, "confidence": confidence})
        return form, confidence


def _chunks(seq: List[int], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
