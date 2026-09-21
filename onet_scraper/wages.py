"""What protects well-paid work: capability, or accountability?

The Frey & Osborne era found automation risk falling as wages rose - the
threatened work was routine and low-paid. Testing the same relationship against
LLM exposure gives a different answer, and the two halves of it disagree in a
way worth keeping rather than smoothing over:

  Across the 195 SOC codes, wage barely correlates with exposure at all
  (r = 0.03). Capability does not care what a job pays.

  Wage correlates with anchoring (r = 0.40). Higher-paid work carries more
  accountability, more consequence, more of the relationship being the point.

  Net susceptibility therefore drifts slightly DOWN with wage across occupations
  (r = -0.18) - but weighted by headcount it rises across wage deciles, because
  employment is not evenly spread across those occupations.

Both are true. The occupation-level correlation answers "does a better-paid job
tend to be safer"; the employment-weighted decile answers "are better-paid
workers safer". Reporting only one of them would be a choice about which
question to answer, so this module reports both.

One thing a correlation cannot do, and the `regress` stage does: exposure and
anchoring correlate at -0.64, so the near-zero wage/exposure correlation above
is partly the two axes cancelling. Holding anchoring and Job Zone fixed, the
exposure coefficient on log wage turns positive and significant - better-paid
work is MORE exposed, conditionally. See `regress.wage_fits`.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Sequence

log = logging.getLogger(__name__)

DECILE_COLUMNS = ("decile", "workers", "mean_wage", "susceptibility", "exposure",
                  "anchoring", "wage_bill_usd")
WAGE_COLUMNS = ("soc_code", "soc_title", "total_employment", "annual_mean_wage",
                "susceptibility", "exposure", "anchoring", "quadrant",
                "protected_by")


def _f(row: dict[str, Any], key: str, default: float | None = 0.0) -> float | None:
    try:
        return float(row.get(key))
    except (TypeError, ValueError):
        return default


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 3:
        return 0.0
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else 0.0


def classify_protection(exposure: float, anchoring: float,
                        x_med: float, a_med: float) -> str:
    """Why is this occupation where it is - capability, or permission?"""
    if exposure < x_med and anchoring < a_med:
        return "Neither - simply less exposed"
    if exposure < x_med:
        return "Capability - AI cannot do much of it"
    if anchoring >= a_med:
        return "Accountability - AI could, a human must answer"
    return "Unprotected - exposed and lightly anchored"


def build(soc_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in soc_rows
            if r.get("total_employment") and r.get("annual_mean_wage")]
    if not rows:
        return {"available": False, "summary": {}, "deciles": [], "detail": []}

    wages = [_f(r, "annual_mean_wage") for r in rows]
    susc = [_f(r, "susceptibility") for r in rows]
    expo = [_f(r, "exposure") for r in rows]
    anch = [_f(r, "anchoring") for r in rows]
    x_med, a_med = statistics.median(expo), statistics.median(anch)

    detail = []
    for r in rows:
        e, a = _f(r, "exposure"), _f(r, "anchoring")
        detail.append({
            "soc_code": r["soc_code"], "soc_title": r["soc_title"],
            "total_employment": _f(r, "total_employment"),
            "annual_mean_wage": _f(r, "annual_mean_wage"),
            "susceptibility": _f(r, "susceptibility"),
            "exposure": e, "anchoring": a,
            "quadrant": r.get("quadrant", ""),
            "protected_by": classify_protection(e, a, x_med, a_med),
        })

    # Employment-weighted wage deciles. An occupation's workers are SPLIT across
    # bucket boundaries rather than assigned whole to whichever bucket its
    # cumulative total lands in. Registered nurses alone are 16% of these
    # workers - larger than a decile - so assigning whole occupations produces
    # buckets ranging from 0.4M to 3.8M and the word "decile" stops being true.
    ordered = sorted(rows, key=lambda r: _f(r, "annual_mean_wage"))
    total = sum(_f(r, "total_employment") for r in ordered)
    step = total / 10
    buckets: list[list[tuple[dict[str, Any], float]]] = [[] for _ in range(10)]
    cursor = 0.0
    for r in ordered:
        remaining = _f(r, "total_employment")
        while remaining > 1e-9:
            b = min(9, int(cursor / step))
            room = (b + 1) * step - cursor
            take = min(remaining, room) if room > 1e-9 else remaining
            buckets[b].append((r, take))
            cursor += take
            remaining -= take

    deciles = []
    for i, group in enumerate(buckets, 1):
        if not group:
            continue
        emp = sum(w for _, w in group)
        wmean = lambda k: sum(_f(r, k) * w for r, w in group) / emp
        deciles.append({
            "decile": i, "workers": round(emp),
            "mean_wage": round(wmean("annual_mean_wage")),
            "susceptibility": round(wmean("susceptibility"), 1),
            "exposure": round(wmean("exposure"), 1),
            "anchoring": round(wmean("anchoring"), 1),
            "wage_bill_usd": round(sum(_f(r, "annual_mean_wage") * w for r, w in group)),
        })

    by_protection: dict[str, dict[str, float]] = {}
    for d in detail:
        b = by_protection.setdefault(d["protected_by"], {"occupations": 0, "workers": 0.0})
        b["occupations"] += 1
        b["workers"] += d["total_employment"]
    for b in by_protection.values():
        b["share_of_workers"] = round(b["workers"] / total, 4)

    summary = {
        "available": True,
        "soc_codes": len(rows),
        "workers": round(total),
        "wage_vs_exposure": round(pearson(wages, expo), 3),
        "wage_vs_anchoring": round(pearson(wages, anch), 3),
        "wage_vs_susceptibility": round(pearson(wages, susc), 3),
        "decile_1_susceptibility": deciles[0]["susceptibility"] if deciles else None,
        "decile_10_susceptibility": deciles[-1]["susceptibility"] if deciles else None,
        "by_protection": by_protection,
        "note": ("Occupation-level correlation and employment-weighted deciles answer "
                 "different questions and point different ways; both are reported."),
    }
    return {"available": True, "summary": summary,
            "deciles": deciles, "detail": detail}
