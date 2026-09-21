"""The network and scoring stages, which run on top of a built dataset."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .build import COLUMNS, append_sqlite, read_table, write_csv
from .config import Settings
from .network import (
    backbone,
    build_incidence,
    force_layout,
    occupation_edges,
    occupation_nodes,
    subtask_network,
    validate_against_related,
    write_graphml,
)
from .score import (
    RUBRIC_VERSION,
    estimate_cost,
    propagate,
    rubric_fingerprint,
    score_subtasks,
    subtask_catalogue,
)

log = logging.getLogger(__name__)



def _optional_ratings(settings: Settings) -> dict[str, Any]:
    """Load the opt-in bulk files, if --with-ratings ever fetched them.

    Absent, every caller falls back to what it did before: the three-term
    tractability average and Job Zone for training depth. Returning empty dicts
    rather than raising is deliberate - the pipeline has to run for someone who
    has not spent the 30 MB.
    """
    import csv as _csv

    from .onet_ratings import (
        education_depth,
        occupation_recurrence,
        rating_precision,
        recurrence,
    )

    bulk = settings.raw_dir / "bulk"
    out: dict[str, Any] = {"recurrence": {}, "education": {}, "precision": {}}

    ratings_path = bulk / "task_ratings.csv"
    if ratings_path.exists():
        rows = list(_csv.DictReader(ratings_path.open()))
        importance = {(r["O*NET-SOC Code"], r["Task ID"]): float(r["Data Value"] or 0)
                      for r in rows if r.get("Scale ID") == "IM"}
        out["recurrence"] = occupation_recurrence(recurrence(rows), importance)
        out["precision"] = rating_precision(rows)
        log.info("task frequency (FT) available for %d occupations",
                 len(out["recurrence"]))

    edu_path = bulk / "education.csv"
    if edu_path.exists():
        depth = education_depth(list(_csv.DictReader(edu_path.open())))
        out["education"] = {c: v["depth"] for c, v in depth.items()}
        out["education_years"] = {c: v["years"] for c, v in depth.items()}
        log.info("education distribution available for %d occupations",
                 len(out["education"]))
    return out


def run_network(settings: Settings, *, min_shared: int, min_co_occurring: int,
                exclude_soc: tuple[str, ...]) -> dict[str, Any]:
    tasks = read_table(settings.out_dir, "tasks")
    task_subtasks = read_table(settings.out_dir, "task_subtasks")
    hierarchy = read_table(settings.out_dir, "subtask_hierarchy")
    occupations = {o["onet_soc_code"]: o for o in read_table(settings.out_dir, "occupations")}
    related = read_table(settings.out_dir, "related_occupations")

    if not task_subtasks:
        raise SystemExit("no task_subtasks table - run the build stage with the bulk crosswalk first")
    if exclude_soc:
        log.info("excluding SOC prefixes %s from the network", ", ".join(exclude_soc))

    occ_dwa, weights, dwa_occ = build_incidence(task_subtasks, tasks, exclude_soc)
    edges = occupation_edges(occ_dwa, weights, occupations, min_shared=min_shared)
    nodes = occupation_nodes(occ_dwa, weights, edges, occupations)

    # Precompute the drawing here: it is O(n^2) per iteration and static, so
    # doing it in the browser just blocks the page on load.
    bone = backbone(edges)
    log.info("laying out %d nodes over a %d-edge backbone", len(nodes), len(bone))
    positions = force_layout([n["onet_soc_code"] for n in nodes], bone)
    for node in nodes:
        x, y = positions.get(node["onet_soc_code"], (0.5, 0.5))
        node["layout_x"], node["layout_y"] = x, y
    dwa_edges, dwa_nodes = subtask_network(dwa_occ, hierarchy, task_subtasks,
                                           min_shared=min_co_occurring)

    tables = {
        "network_occupation_edges": edges,
        "network_occupation_nodes": nodes,
        "network_subtask_edges": dwa_edges,
        "network_subtask_nodes": dwa_nodes,
    }
    for name, rows in tables.items():
        write_csv(settings.out_dir / f"{name}.csv", rows, COLUMNS[name])
    append_sqlite(settings.out_dir / "onet_stem.sqlite", tables)

    write_graphml(settings.out_dir / "network_occupations.graphml", nodes, edges,
                  "onet_soc_code", "cosine")
    write_graphml(settings.out_dir / "network_subtasks.graphml", dwa_nodes, dwa_edges,
                  "dwa_id", "co_occurring_occupations")

    check = validate_against_related(edges, related)
    report = {
        "occupation_nodes": len(nodes),
        "occupation_edges": len(edges),
        "subtask_nodes": len(dwa_nodes),
        "subtask_edges": len(dwa_edges),
        "density": round(2 * len(edges) / (len(nodes) * (len(nodes) - 1)), 4) if len(nodes) > 1 else 0,
        "min_shared_subtasks": min_shared,
        "excluded_soc_prefixes": list(exclude_soc),
        "validation_vs_onet_related": check,
    }
    (settings.out_dir / "network_report.json").write_text(json.dumps(report, indent=2))
    for key, value in report.items():
        log.info("%-28s %s", key, value)
    if check.get("available"):
        log.info("recall@%d vs O*NET's own related list: %.1f%% "
                 "(ceiling %.1f%%, so %.1f%% of what is reachable)",
                 check["k"], 100 * check["recall_at_k"],
                 100 * check["max_achievable_recall_at_k"],
                 100 * check["recall_vs_ceiling"])
    else:
        log.info("no related_occupations table - run fetch-descriptors to enable validation")
    return report


def run_score(settings: Settings, *, model: str, chunk_size: int, workers: int,
              refresh: bool, dry_run: bool) -> dict[str, Any]:
    task_subtasks = read_table(settings.out_dir, "task_subtasks")
    hierarchy = read_table(settings.out_dir, "subtask_hierarchy")
    tasks = read_table(settings.out_dir, "tasks")
    occupations = read_table(settings.out_dir, "occupations")
    if not task_subtasks:
        raise SystemExit("no task_subtasks table - run the build stage first")

    catalogue = subtask_catalogue(task_subtasks, hierarchy)
    estimate = estimate_cost(catalogue, chunk_size, model)
    log.info("rubric %s (fingerprint %s)", RUBRIC_VERSION, rubric_fingerprint())
    for key, value in estimate.items():
        log.info("  %-22s %s", key, value)
    if dry_run:
        log.info("dry run: nothing sent to the API")
        return {"estimate": estimate, "dry_run": True}

    scores, failures = score_subtasks(catalogue, settings.raw_dir, model=model,
                                      chunk_size=chunk_size, workers=workers,
                                      refresh=refresh)
    if failures:
        (settings.raw_dir / "score_failures.json").write_text(json.dumps(failures, indent=2))
        log.error("%d subtasks failed to score; re-run to retry only those", len(failures))

    task_scores, occ_scores = propagate(scores, tasks, task_subtasks, occupations)
    tables = {
        "subtask_automation_scores": scores,
        "task_automation_scores": task_scores,
        "occupation_automation_scores": occ_scores,
    }
    for name, rows in tables.items():
        write_csv(settings.out_dir / f"{name}.csv", rows, COLUMNS[name])
    append_sqlite(settings.out_dir / "onet_stem.sqlite", tables)

    report = {
        "rubric_version": RUBRIC_VERSION,
        "rubric_fingerprint": rubric_fingerprint(),
        "model": model,
        "subtasks_scored": len(scores),
        "subtasks_requested": len(catalogue),
        "tasks_scored": len(task_scores),
        "occupations_scored": len(occ_scores),
        "failures": len(failures),
        "estimate": estimate,
    }
    (settings.out_dir / "scoring_report.json").write_text(json.dumps(report, indent=2))
    log.info("scored %d/%d subtasks -> %d tasks -> %d occupations",
             len(scores), len(catalogue), len(task_scores), len(occ_scores))
    return report


def run_report(settings: Settings) -> dict[str, Any]:
    """Compute the susceptibility index and render the dashboard."""
    from .dashboard import build_dashboard
    from .susceptibility import (
        OCC_SUSC_COLUMNS,
        SUBTASK_SUSC_COLUMNS,
        TASK_SUSC_COLUMNS,
        build,
        convergent_validity,
    )

    subtask_scores = read_table(settings.out_dir, "subtask_automation_scores")
    if not subtask_scores:
        raise SystemExit("no subtask_automation_scores - run the score stage first")
    task_scores = read_table(settings.out_dir, "task_automation_scores")
    occ_scores = read_table(settings.out_dir, "occupation_automation_scores")
    task_subtasks = read_table(settings.out_dir, "task_subtasks")
    occupations = read_table(settings.out_dir, "occupations")

    result = build(subtask_scores, task_scores, occ_scores, task_subtasks, occupations)
    splits = result.pop("_splits")[0]

    columns = {"subtask_susceptibility": SUBTASK_SUSC_COLUMNS,
               "task_susceptibility": TASK_SUSC_COLUMNS,
               "occupation_susceptibility": OCC_SUSC_COLUMNS}
    for name, rows in result.items():
        COLUMNS[name] = columns[name]
        write_csv(settings.out_dir / f"{name}.csv", rows, columns[name])
    append_sqlite(settings.out_dir / "onet_stem.sqlite", result)

    # Watson handoff framing, computed from the same scored dimensions.
    from .handoff import HANDOFF_COLUMNS, build as build_handoff, summarise as sum_handoff
    occ_scores = read_table(settings.out_dir, "occupation_automation_scores")
    soc_emp = read_table(settings.out_dir, "soc_susceptibility")
    emp_by_onet: dict[str, float] = {}
    for srow in soc_emp:
        for code in (srow.get("onet_codes") or "").split(";"):
            if code and srow.get("total_employment"):
                # SOC employment, carried for scale only - never summed across
                # the O*NET occupations inside one SOC.
                emp_by_onet[code] = float(srow["total_employment"])
    occ_titles = {o["onet_soc_code"]: o for o in occupations}
    for row in occ_scores:
        meta = occ_titles.get(row["onet_soc_code"], {})
        row.setdefault("title", meta.get("title", ""))
        row["stem_occupation_types"] = meta.get("stem_occupation_types", "")
    extra = _optional_ratings(settings)
    # Recurrence is reported, not folded into tractability: it is orthogonal to
    # the existing terms, so averaging it in compresses the axis by a third and
    # invalidates the frontier calibration. See handoff.axes().
    handoff_rows = build_handoff(occ_scores, employment=emp_by_onet)
    for row in handoff_rows:
        row["recurrence"] = extra["recurrence"].get(row["onet_soc_code"])
    COLUMNS["occupation_handoff"] = HANDOFF_COLUMNS
    write_csv(settings.out_dir / "occupation_handoff.csv", handoff_rows, HANDOFF_COLUMNS)
    append_sqlite(settings.out_dir / "onet_stem.sqlite",
                  {"occupation_handoff": handoff_rows})
    # The SOC map, so the summary's employment figures collapse before summing.
    soc_of_h: dict[str, str] = {}
    for r in read_table(settings.out_dir, "soc_susceptibility"):
        for c in (r.get("onet_codes") or "").split(";"):
            if c:
                soc_of_h[c] = r["soc_code"]
    handoff_summary = sum_handoff(handoff_rows, soc_of_h)
    (settings.out_dir / "handoff_report.json").write_text(
        json.dumps(handoff_summary, indent=2))
    log.info("handoff framing: %s", handoff_summary["by_classification"])
    if handoff_summary.get("employment_share_at_watch_points") is not None:
        log.info("  %.1f%% of workers are at a watch point",
                 100 * handoff_summary["employment_share_at_watch_points"])
    log.info("  %d occupations have a pending crossing, %d of them at one of the "
             "two weighty crossings", handoff_summary["with_pending_crossing"],
             handoff_summary["at_a_weighty_crossing"])

    validity = convergent_validity(result["subtask_susceptibility"])
    log.info("convergent validity (mean index by the model's own verdict): %s", validity)
    if not validity.get("monotonic"):
        log.warning("index does NOT order the verdicts monotonically - treat it with suspicion")

    scoring_meta = {}
    report_path = settings.out_dir / "scoring_report.json"
    if report_path.exists():
        scoring_meta = json.loads(report_path.read_text())

    # Optional: the employment layer, if the employment stage has been run.
    soc_rows = read_table(settings.out_dir, "soc_susceptibility")
    employment_meta: dict[str, Any] = {}
    emp_report = settings.out_dir / "employment_report.json"
    if emp_report.exists():
        payload = json.loads(emp_report.read_text())
        employment_meta = dict(payload.get("headline", {}))
        employment_meta["release"] = payload.get("oews_release", "")

    dashboard = build_dashboard(
        settings.out_dir / "dashboard.html",
        result["occupation_susceptibility"],
        result["task_susceptibility"],
        result["subtask_susceptibility"],
        splits,
        {"model": scoring_meta.get("model", ""),
         "rubric_version": scoring_meta.get("rubric_version", ""),
         # written by the consolidate stage; the display surfaces say "scored
         # by X" and X stopped being one model
         "provenance": scoring_meta.get("provenance", "")},
        soc=soc_rows,
        employment=employment_meta,
        benchmarks=read_table(settings.out_dir, "external_benchmarks"),
        uncertainty=read_table(settings.out_dir, "occupation_uncertainty"),
        net_edges=read_table(settings.out_dir, "network_occupation_edges"),
        net_nodes=read_table(settings.out_dir, "network_occupation_nodes"),
        handoff=handoff_rows,
        dimensions=read_table(settings.out_dir, "occupation_automation_scores"),
    )

    report = {"splits": splits, "convergent_validity": validity,
              "handoff": handoff_summary,
              "rows": {k: len(v) for k, v in result.items()},
              "dashboard": str(dashboard)}
    (settings.out_dir / "susceptibility_report.json").write_text(json.dumps(report, indent=2))
    log.info("open %s", dashboard)
    return report


def run_employment(settings: Settings, client) -> dict[str, Any]:
    """Join BLS OEWS employment and wages at the SOC level."""
    from .employment import (
        OEWS_COLUMNS,
        SOC_SUSC_COLUMNS,
        build_soc_table,
        fetch_oews,
        headline_stats,
    )

    occ = read_table(settings.out_dir, "occupation_susceptibility")
    if not occ:
        raise SystemExit("no occupation_susceptibility - run the report stage first")

    oews = fetch_oews(client, settings)
    soc_rows, diagnostics = build_soc_table(occ, oews["rows"])

    matched = {r["soc_code"] for r in soc_rows}
    tables = {
        "oews_occupations": [v for k, v in sorted(oews["rows"].items()) if k in matched],
        "soc_susceptibility": soc_rows,
    }
    COLUMNS["oews_occupations"] = OEWS_COLUMNS
    COLUMNS["soc_susceptibility"] = SOC_SUSC_COLUMNS
    for name, rows in tables.items():
        write_csv(settings.out_dir / f"{name}.csv", rows, COLUMNS[name])
    append_sqlite(settings.out_dir / "onet_stem.sqlite", tables)

    stats = headline_stats(soc_rows)
    report = {"oews_release": f"May 20{oews['release']}", "join": diagnostics,
              "headline": stats}
    (settings.out_dir / "employment_report.json").write_text(json.dumps(report, indent=2))

    log.info("-" * 68)
    log.info("%d O*NET occupations collapsed to %d SOC codes (%d SOCs had >1 detail "
             "occupation, which a naive join would have multi-counted)",
             diagnostics["onet_occupations_in"], diagnostics["soc_codes_out"],
             diagnostics["collapsed_socs"])
    if diagnostics["unmatched_onet_codes"]:
        log.warning("%d O*NET codes had no OEWS match: %s",
                    len(diagnostics["unmatched_onet_codes"]),
                    diagnostics["unmatched_onet_codes"][:6])
    log.info("total employment covered: %s", f"{stats['total_employment']:,}")
    log.info("employment-weighted susceptibility: %s (unweighted %s, shift %+.1f)",
             stats["employment_weighted_susceptibility"],
             stats["unweighted_susceptibility"], stats["weighting_shifts_result_by"])
    log.info("workers in high-susceptibility occupations: %s (%.1f%%)",
             f"{stats['workers_in_high_susceptibility_occupations']:,}",
             100 * stats["share_workers_high_susceptibility"])
    for q, v in sorted(stats["by_quadrant"].items(), key=lambda kv: -kv[1]["employment"]):
        log.info("  %-16s %12s workers (%.1f%%)  $%.0fB wage bill",
                 q, f"{v['employment']:,.0f}", 100 * v["share_of_employment"],
                 v["wage_bill_usd"] / 1e9)
    return report


def run_external(settings: Settings, client) -> dict[str, Any]:
    """Benchmark the index against published exposure measures, incl. human ratings."""
    from .external import (
        BENCHMARK_COLUMNS,
        CITATION,
        correlate,
        fetch_benchmarks,
        join,
    )

    ours = read_table(settings.out_dir, "occupation_susceptibility")
    if not ours:
        raise SystemExit("no occupation_susceptibility - run the report stage first")

    benchmarks = fetch_benchmarks(client, settings)
    rows, coverage = join(ours, benchmarks)
    COLUMNS["external_benchmarks"] = BENCHMARK_COLUMNS
    write_csv(settings.out_dir / "external_benchmarks.csv", rows, BENCHMARK_COLUMNS)
    append_sqlite(settings.out_dir / "onet_stem.sqlite", {"external_benchmarks": rows})

    by_index = correlate(rows, "our_susceptibility")
    by_exposure = correlate(rows, "our_exposure")
    report = {"citation": CITATION, "coverage": coverage,
              "susceptibility_vs": by_index, "exposure_vs": by_exposure}
    (settings.out_dir / "external_validation.json").write_text(json.dumps(report, indent=2))

    log.info("-" * 74)
    log.info("matched %d of %d occupations against published measures",
             coverage["matched"], len(ours))
    log.info("%-22s %5s %8s %9s   %s", "measure", "n", "pearson", "spearman", "expectation")
    for c in by_index:
        log.info("%-22s %5d %8s %9s   %s", c["measure"], c["n"],
                 c["pearson"], c["spearman"], c["expectation"])
    human = [c for c in by_index if c["measure"].startswith("human") and c["pearson"]]
    if human:
        best = max(human, key=lambda c: c["pearson"])
        log.info("-" * 74)
        log.info("strongest agreement with HUMAN expert ratings: %s r=%.3f (n=%d)",
                 best["measure"], best["pearson"], best["n"])
        log.info("this is the external check - the rest of the pipeline is self-consistent "
                 "by construction, but this is not")
    return report


def run_figures(settings: Settings, *, chrome: str | None, dark: bool,
                scale: int, only: tuple[str, ...]) -> list[Path]:
    """Export each dashboard chart as a PNG."""
    from .figures import render

    dashboard = settings.out_dir / "dashboard.html"
    if not dashboard.exists():
        raise SystemExit("no dashboard.html - run the report stage first")
    out_dir = settings.out_dir / "figures"
    written = render(dashboard, out_dir, chrome=chrome, dark=dark, scale=scale, only=only)
    log.info("%d figure(s) in %s", len(written), out_dir)
    return written


def run_scroller(settings: Settings) -> Path:
    """Render the scroll-driven narrative from the built tables."""
    from .scroller import build_payload, build_scroller

    occ = read_table(settings.out_dir, "occupation_susceptibility")
    handoff = read_table(settings.out_dir, "occupation_handoff")
    benchmarks = read_table(settings.out_dir, "external_benchmarks")
    soc = read_table(settings.out_dir, "soc_susceptibility")
    if not (occ and handoff):
        raise SystemExit("run the report stage first (and score before that)")
    for name, table in (("external_benchmarks", benchmarks), ("soc_susceptibility", soc)):
        if not table:
            log.warning("%s missing - that chapter will be thin", name)

    emp_path = settings.out_dir / "employment_report.json"
    emp = json.loads(emp_path.read_text()) if emp_path.exists() else {}
    score_path = settings.out_dir / "scoring_report.json"
    meta = json.loads(score_path.read_text()) if score_path.exists() else {}

    tasks = read_table(settings.out_dir, "task_susceptibility")
    subtasks = read_table(settings.out_dir, "subtask_susceptibility")
    links = read_table(settings.out_dir, "task_subtasks")
    pw_path = settings.out_dir / "pathways_report.json"
    pathways = json.loads(pw_path.read_text()) if pw_path.exists() else {}
    if pathways:
        pathways["deciles"] = read_table(settings.out_dir, "wage_deciles")
        moves = read_table(settings.out_dir, "transitions")
        stranded = read_table(settings.out_dir, "stranded_occupations")
        # a handful of each, biggest first, for the figure
        from .scroller import _clip
        pathways["moveSample"] = [
            {"t": _clip(m["title"], 28), "d": _clip(m["destination"], 28),
             "s": float(m["susceptibility"]), "ds": float(m["destination_susceptibility"]),
             "v": m["verdict"]}
            for m in moves[:6]]
        pathways["strandedSample"] = [
            {"t": _clip(r["title"], 28), "s": float(r["susceptibility"])}
            for r in stranded[:6]]
    ch_path = settings.out_dir / "churn_report.json"
    churn = json.loads(ch_path.read_text()) if ch_path.exists() else {}
    if churn:
        churn["steps"] = read_table(settings.out_dir, "task_churn_steps")
        rows = [r for r in read_table(settings.out_dir, "task_churn_occupations")
                if r.get("susceptibility") and r.get("last_reviewed")
                and int(r["last_reviewed"]) >= churn.get("review_cutoff", 2022)
                and int(r["tasks_first"]) >= 8]
        churn["byExposure"] = []
        for label, lo, hi in (("Highly exposed", 65, 200), ("Middling", 55, 65),
                              ("Low exposure", 0, 55)):
            g = [r for r in rows if lo <= float(r["susceptibility"]) < hi]
            if g:
                churn["byExposure"].append({
                    "label": label, "n": len(g),
                    "turnover": round(sum(float(r["turnover_rate"]) for r in g)/len(g), 4)})
    sc_path = settings.out_dir / "scenarios_report.json"
    scen = json.loads(sc_path.read_text()) if sc_path.exists() else {}
    emerging = read_table(settings.out_dir, "emerging_tasks")
    payload = build_payload(occ, tasks, subtasks, links, handoff, benchmarks, soc, emp,
                            pathways, churn, scen, emerging)
    return build_scroller(settings.out_dir / "story.html", payload, meta)


def run_churn(settings: Settings, client, releases: tuple[str, ...] = ()) -> dict[str, Any]:
    """Diff task statements across archived O*NET releases."""
    from .churn import CHURN_COLUMNS, DEFAULT_RELEASES, OCC_CHURN_COLUMNS, build

    occ = read_table(settings.out_dir, "occupation_susceptibility")
    if not occ:
        raise SystemExit("run the report stage first")
    titles = {o["onet_soc_code"]: o["title"] for o in occ}
    susc = {o["onet_soc_code"]: float(o["susceptibility"]) for o in occ}

    steps, occ_rows, summary = build(
        client, releases or DEFAULT_RELEASES, titles.keys(), titles, susc)

    COLUMNS["task_churn_steps"] = CHURN_COLUMNS
    COLUMNS["task_churn_occupations"] = OCC_CHURN_COLUMNS
    tables = {"task_churn_steps": steps, "task_churn_occupations": occ_rows}
    for name, rows in tables.items():
        write_csv(settings.out_dir / f"{name}.csv", rows, COLUMNS[name])
    append_sqlite(settings.out_dir / "onet_stem.sqlite", tables)
    (settings.out_dir / "churn_report.json").write_text(json.dumps(summary, indent=2))

    log.info("-" * 70)
    log.info("task churn %s (%s)", summary["span"], summary["years"])
    log.info("  %d occupations compared, %d skipped on taxonomy change",
             summary["occupations_compared"], summary["occupations_skipped"])
    log.info("  %d tasks -> %d tasks  (%d added, %d retired, %d survived)",
             summary["tasks_start"], summary["tasks_end"],
             summary["added"], summary["retired"], summary["survived"])
    log.info("  turnover %.1f%% · %.1f%% of today's tasks did not exist at the start",
             100*summary["turnover_rate"], 100*summary["added_rate"])
    f, st = summary["refreshed_only"], summary["stale_only"]
    log.info("  split on whether O*NET re-surveyed the occupation since %d:",
             summary["review_cutoff"])
    log.info("    re-surveyed    %3d occupations - turnover %.1f%%, %.1f%% of tasks new",
             summary["reviewed_since_cutoff"], 100*f["turnover_rate"], 100*f["added_rate"])
    log.info("    never looked at %3d occupations - turnover %.1f%%, %.1f%% new",
             summary["not_reviewed_since_cutoff"], 100*st["turnover_rate"],
             100*st["added_rate"])
    log.info("  the overall figure is diluted by occupations nobody checked")
    return summary


def run_pathways(settings: Settings) -> dict[str, Any]:
    """Wage protection analysis and the transition map. Both run on local tables."""
    from .transitions import (
        STRANDED_COLUMNS,
        TRANSITION_COLUMNS,
        build as build_moves,
        sweep,
    )
    from .wages import DECILE_COLUMNS, WAGE_COLUMNS, build as build_wages

    occ = read_table(settings.out_dir, "occupation_susceptibility")
    soc = read_table(settings.out_dir, "soc_susceptibility")
    edges = read_table(settings.out_dir, "network_occupation_edges")
    links = read_table(settings.out_dir, "task_subtasks")
    subs = read_table(settings.out_dir, "subtask_susceptibility")
    if not (occ and edges and links):
        raise SystemExit("run report and network first")

    tables: dict[str, list[dict[str, Any]]] = {}
    report: dict[str, Any] = {}

    # --- 1. what protects well-paid work ---------------------------------
    if soc:
        w = build_wages(soc)
        if w.get("available"):
            COLUMNS["wage_deciles"] = DECILE_COLUMNS
            COLUMNS["wage_protection"] = WAGE_COLUMNS
            tables["wage_deciles"] = w["deciles"]
            tables["wage_protection"] = w["detail"]
            report["wages"] = w["summary"]
            s = w["summary"]
            log.info("-" * 70)
            log.info("wage vs exposure r=%s · vs anchoring r=%s · vs susceptibility r=%s",
                     s["wage_vs_exposure"], s["wage_vs_anchoring"],
                     s["wage_vs_susceptibility"])
            log.info("susceptibility, bottom wage decile %s → top %s",
                     s["decile_1_susceptibility"], s["decile_10_susceptibility"])
            for k, v in sorted(s["by_protection"].items(),
                               key=lambda kv: -kv[1]["workers"]):
                log.info("  %-44s %2d occs · %5.1f%% of workers",
                         k, v["occupations"], 100*v["share_of_workers"])
    else:
        log.warning("no soc_susceptibility - run the employment stage for the wage half")

    # --- 2. where the people could go ------------------------------------
    occ_dwa: dict[str, set[str]] = {}
    for l in links:
        if l.get("dwa_id"):
            occ_dwa.setdefault(l["onet_soc_code"], set()).add(l["dwa_id"])
    dwa_susc = {s["dwa_id"]: float(s["susceptibility"]) for s in subs}
    employment: dict[str, float] = {}
    for r in soc:
        if r.get("total_employment"):
            for c in (r.get("onet_codes") or "").split(";"):
                if c:
                    employment[c] = float(r["total_employment"])

    moves, stranded, msum = build_moves(edges, occ, occ_dwa, dwa_susc, employment)
    COLUMNS["transitions"] = TRANSITION_COLUMNS
    COLUMNS["stranded_occupations"] = STRANDED_COLUMNS
    tables["transitions"] = moves
    tables["stranded_occupations"] = stranded
    msum["sensitivity"] = sweep(edges, occ, occ_dwa, dwa_susc, employment)
    report["transitions"] = msum

    log.info("-" * 70)
    log.info("%d of %d occupations have a plausible less-exposed destination; "
             "%d are stranded (%.0f%%)", msum["with_a_destination"], msum["occupations"],
             msum["stranded"], 100*msum["stranded_share"])
    log.info("  of the moves that exist, %d are real and %d carry the exposure along",
             msum["real_moves"], msum["carries_exposure"])
    log.info("  %d exposed occupations are stranded (%s workers)",
             msum["exposed_and_stranded"],
             f"{msum['exposed_and_stranded_workers']:,}")
    log.info("  sensitivity: stranded ranges %d-%d across the threshold sweep",
             min(s["stranded"] for s in msum["sensitivity"]),
             max(s["stranded"] for s in msum["sensitivity"]))

    for name, rows in tables.items():
        write_csv(settings.out_dir / f"{name}.csv", rows, COLUMNS[name])
    append_sqlite(settings.out_dir / "onet_stem.sqlite", tables)
    (settings.out_dir / "pathways_report.json").write_text(json.dumps(report, indent=2))
    return report


def run_publish(settings: Settings, domain: str | None = None) -> Path:
    """Assemble docs/ for GitHub Pages."""
    from .publish import DOMAIN, publish

    return publish(settings.out_dir, Path("docs"), domain or DOMAIN)


def run_scenarios(settings: Settings) -> dict[str, Any]:
    """Classify every task under each scenario and compute the employment flows."""
    from .scenarios import (
        OCC_SCENARIO_COLUMNS,
        TASK_FATE_COLUMNS,
        by_occupation,
        flows,
        summarise,
        task_fates,
    )

    tasks = read_table(settings.out_dir, "task_susceptibility")
    if not tasks:
        raise SystemExit("run the report stage first")
    occ = read_table(settings.out_dir, "occupation_susceptibility")
    titles = {o["onet_soc_code"]: o["title"] for o in occ}
    emerging = read_table(settings.out_dir, "emerging_tasks")
    soc = read_table(settings.out_dir, "soc_susceptibility")
    employment: dict[str, float] = {}
    for r in soc:
        if r.get("total_employment"):
            for c in (r.get("onet_codes") or "").split(";"):
                if c:
                    employment[c] = float(r["total_employment"])
    destinations = {m["onet_soc_code"] for m in read_table(settings.out_dir, "transitions")}
    soc_of: dict[str, str] = {}
    for r in soc:
        for c in (r.get("onet_codes") or "").split(";"):
            if c:
                soc_of[c] = r["soc_code"]

    fates = task_fates(tasks)
    occ_rows = by_occupation(fates, emerging, titles, employment, destinations)
    report = summarise(fates, occ_rows, len(emerging))
    report["flows"] = flows(occ_rows, soc_of)

    COLUMNS["task_fates"] = TASK_FATE_COLUMNS
    COLUMNS["occupation_scenarios"] = OCC_SCENARIO_COLUMNS
    tables = {"task_fates": fates, "occupation_scenarios": occ_rows}
    for name, rows in tables.items():
        write_csv(settings.out_dir / f"{name}.csv", rows, COLUMNS[name])
    append_sqlite(settings.out_dir / "onet_stem.sqlite", tables)
    (settings.out_dir / "scenarios_report.json").write_text(json.dumps(report, indent=2))

    log.info("-" * 72)
    log.info("%d tasks classified under 3 scenarios · %d new tasks observed",
             report["tasks"], report["new_tasks_observed"])
    for name, s in report["scenarios"].items():
        log.info("  %-12s automated %4d (%3.0f%%)  augmented %4d  unchanged %4d  "
                 "· %3d occupations reshaped", s["label"], s["automated"],
                 100*s["share_automated"], s["augmented"], s["unchanged"],
                 s["occupations_reshaped"])
    log.info("  employment-weighted:")
    for name, fl in report["flows"].items():
        log.info("    %-12s %5.1f%% of workers in reshaped occupations — "
                 "%4.1f%% could move, %4.1f%% stranded",
                 report["scenarios"][name]["label"], 100*fl["share_reshaped"],
                 100*fl["share_with_destination"], 100*fl["share_stranded"])
    return report


def run_security(settings: Settings) -> dict[str, Any]:
    """Place every occupation on the efficiency / risk / reconstitution matrix."""
    from .security import (
        FIELD_COLUMNS,
        SECURITY_COLUMNS,
        build,
        by_field,
        field_map,
        summarise,
    )

    fates = read_table(settings.out_dir, "task_fates")
    if not fates:
        raise SystemExit("run the scenarios stage first")
    task_scores = read_table(settings.out_dir, "task_automation_scores")
    occ = read_table(settings.out_dir, "occupation_susceptibility")
    net = read_table(settings.out_dir, "network_occupation_nodes")
    similarity = {n["onet_soc_code"]: float(n["mean_similarity"])
                  for n in net if n.get("mean_similarity")}
    hand = {h["onet_soc_code"]: h for h in read_table(settings.out_dir,
                                                      "occupation_handoff")}

    # Employment lives at 6-digit SOC; map it onto every O*NET code under that
    # SOC and keep the reverse map so aggregates can collapse back.
    soc = read_table(settings.out_dir, "soc_susceptibility")
    employment: dict[str, float] = {}
    soc_of: dict[str, str] = {}
    for r in soc:
        for c in (r.get("onet_codes") or "").split(";"):
            if not c:
                continue
            soc_of[c] = r["soc_code"]
            if r.get("total_employment"):
                employment[c] = float(r["total_employment"])

    fields = field_map(read_table(settings.out_dir, "occupation_stem_categories"),
                       read_table(settings.out_dir, "stem_categories"))
    extra = _optional_ratings(settings)
    rows = build(fates, task_scores, occ, employment, similarity, hand,
                 fields, soc_of, extra['education'])
    fields = by_field(rows, soc_of)
    report = summarise(rows, soc_of)

    COLUMNS["security_matrix"] = SECURITY_COLUMNS
    COLUMNS["security_fields"] = FIELD_COLUMNS
    tables = {"security_matrix": rows, "security_fields": fields}
    for name, table in tables.items():
        write_csv(settings.out_dir / f"{name}.csv", table, COLUMNS[name])
    append_sqlite(settings.out_dir / "onet_stem.sqlite", tables)
    (settings.out_dir / "security_report.json").write_text(json.dumps(report, indent=2))

    from .matrix3d import build_matrix3d
    build_matrix3d(rows, fields, settings.out_dir / "security_matrix.html", soc_of)

    log.info("-" * 72)
    log.info("national security matrix · %d occupations x 3 scenarios",
             report["occupations"])
    for scenario, s in report["scenarios"].items():
        trap = s["by_octant"]["Strategic trap"]
        prot = s["by_octant"]["Protect"]
        log.info("  %-12s efficiency %4.1f  risk %4.1f  reconstitution %4.1f",
                 scenario, s["mean_efficiency"], s["mean_removal_risk"],
                 s["mean_reconstitution"])
        log.info("               strategic trap %3d occ (%4.1f%% of workers) · "
                 "protect %3d occ (%4.1f%%)",
                 trap["occupations"], 100 * trap["share_employment"],
                 prot["occupations"], 100 * prot["share_employment"])
    return report


def run_validate_derived(settings: Settings) -> dict[str, Any]:
    """Plausibility checks on the computed measures, not the scrape."""
    from .validate_derived import log_report, validate_derived

    security = read_table(settings.out_dir, "security_matrix")
    if not security:
        raise SystemExit("run the security stage first")
    report_path = settings.out_dir / "security_report.json"
    checks, summary = validate_derived(
        read_table(settings.out_dir, "occupation_susceptibility"),
        read_table(settings.out_dir, "occupation_handoff"),
        security,
        json.loads(report_path.read_text()) if report_path.exists() else {},
        read_table(settings.out_dir, "soc_susceptibility"),
    )
    errors = log_report(checks, summary)
    (settings.out_dir / "validation_derived.json").write_text(
        json.dumps({"checks": checks, "summary": summary}, indent=2))
    if errors:
        log.error("%d derived-measure check(s) failed; see "
                  "data/out/validation_derived.json", errors)
        raise SystemExit(1)
    return {"checks": checks, "summary": summary}


def run_smoke(settings: Settings, *, chrome: str | None = None,
              docs: Path | None = None) -> dict[str, Any]:
    """Load every published page in a browser and assert it rendered."""
    from .figures import find_chrome
    from .smoke import check, log_report

    binary = find_chrome(chrome)
    if not binary:
        raise SystemExit("no Chrome/Chromium found; pass --chrome /path/to/binary")
    target = docs or (Path(__file__).resolve().parents[1] / "docs")
    log.info("smoke-testing %s with %s", target, Path(binary).name)
    results = check(binary, target)
    failed = log_report(results)
    (settings.out_dir / "smoke_report.json").write_text(json.dumps(results, indent=2))
    if failed:
        log.error("%d page assertion(s) failed", failed)
        raise SystemExit(1)
    log.info("%d assertions passed across %d pages", len(results),
             len({r["page"] for r in results}))
    return {"results": results, "failed": failed}


def run_retest(settings: Settings, *, model: str = "claude-sonnet-5",
               chunk_size: int = 0, workers: int = 4,
               budget_usd: float = 4.0, label: str = "") -> dict[str, Any]:
    """Score the catalogue a second time and report agreement with the first.

    Writes into its own checkpoint directory so the original pass is never
    touched - the whole point is to have two independent sets to compare, and a
    run that overwrote the first would destroy the measurement it was made for.

    `budget_usd` is a hard stop, not a warning. An estimate is cheap to get
    wrong and this spends real money.
    """
    from .reliability import compare, log_report
    from .score import estimate_cost, load_scores, score_subtasks, subtask_catalogue

    # Chunk size drives cost as well as latency: the system prompt is re-sent
    # per request, so 12 per chunk costs 81 requests and $3.28 where 25 costs 39
    # and $2.53 for the same 963 subtasks. The retest defaults to the larger
    # chunk rather than inheriting the scoring default.
    chunk_size = chunk_size or 25
    label = label or model
    links = read_table(settings.out_dir, "task_subtasks")
    hierarchy = read_table(settings.out_dir, "subtask_hierarchy")
    if not links:
        raise SystemExit("no task_subtasks table - run the build stage first")
    # Same catalogue the first pass scored, built the same way: a retest that
    # scored a differently-assembled catalogue would not be comparing raters.
    catalogue = subtask_catalogue(links, hierarchy)
    if not catalogue:
        raise SystemExit("no subtask catalogue; run the build stage first")

    first = load_scores(settings.raw_dir)
    if not first:
        raise SystemExit("no first pass to compare against; run the score stage first")

    est = estimate_cost(catalogue, chunk_size, model)
    log.info("retest with %s: %d subtasks, %d requests, estimated $%.2f",
             model, est["subtasks"], est["requests"], est["est_cost_usd"])
    if est["est_cost_usd"] > budget_usd:
        raise SystemExit(
            f"estimated ${est['est_cost_usd']:.2f} exceeds the ${budget_usd:.2f} "
            f"budget for this run; pass --budget to raise it deliberately")

    # One directory per pass, named for the pass rather than shared.
    #
    # It was a single fixed "retest" directory, which was fine for one extra
    # pass and actively destructive for a second: score_subtasks checkpoints by
    # dwa_id, so an Opus run would have overwritten the Sonnet scores key by key
    # and the cross-model comparison they were bought for would have quietly
    # become opus-against-opus. Refusing to reuse a populated slot is the point.
    retest_dir = settings.raw_dir / "retest" / label
    if retest_dir.exists() and (retest_dir / "subtask_scores.jsonl").exists():
        existing = load_scores(retest_dir)
        if existing:
            raise SystemExit(
                f"{retest_dir} already holds {len(existing)} scores. Pass "
                f"--retest-label to name this pass something else (a second run "
                f"of the same model needs its own slot), or delete that "
                f"directory to redo it.")
    retest_dir.mkdir(parents=True, exist_ok=True)
    scores, failures = score_subtasks(catalogue, retest_dir, model=model,
                                      chunk_size=chunk_size, workers=workers)
    if failures:
        # `failures` carries one entry per SUBTASK in a failed chunk, so calling
        # it a chunk count reported 963 failures for 39 failed requests.
        log.warning("%d subtask(s) failed to score", len(failures))

    # A run with no API key reaches this point having scored nothing, and the
    # comparison below will happily report r = 0.000 over n = 0 - a failed run
    # presented as a finished measurement, which is the exact failure mode this
    # project keeps meeting. Refuse to write a report there is no evidence for.
    if not scores:
        raise SystemExit(
            f"the retest scored nothing ({len(failures)} subtask(s) failed). "
            f"No report written. Check ANTHROPIC_API_KEY and the model name.")
    if len(scores) < len(catalogue) * 0.9:
        raise SystemExit(
            f"the retest scored only {len(scores)} of {len(catalogue)} subtasks; "
            f"agreement over a partial catalogue is not comparable to the first "
            f"pass. No report written.")

    kind = "test-retest" if model == first[next(iter(first))].get("model") else "cross-model"
    report = compare(list(first.values()), scores, kind=kind)
    if not report["subtasks_compared"]:
        raise SystemExit("no subtask appears in both passes; nothing to compare")
    report["first_model"] = first[next(iter(first))].get("model")
    report["second_model"] = model
    report["failures"] = len(failures)
    log_report(report)

    (settings.out_dir / "reliability_report.json").write_text(
        json.dumps(report, indent=2))
    COLUMNS["subtask_scores_retest"] = tuple(scores[0].keys()) if scores else ()
    if scores:
        write_csv(settings.out_dir / "subtask_scores_retest.csv", scores,
                  COLUMNS["subtask_scores_retest"])
    return report


def run_consolidate(settings: Settings) -> dict[str, Any]:
    """Replace the single-pass subtask scores with the mean of both passes.

    Everything downstream reads the subtask_automation_scores table, so writing
    the consolidated scores there propagates through the task and occupation
    indices, the scenarios, the handoff axes and the security matrix without
    any of them needing to know two passes exist.

    Both checkpoints are left untouched in data/raw, so this is reversible: the
    single-pass tables can be regenerated from them at any time.
    """
    from .reliability import consolidate, consolidation_summary
    from .score import load_scores

    first = load_scores(settings.raw_dir)
    if not first:
        raise SystemExit("no first pass; run the score stage first")

    # Every pass under data/raw/retest/, discovered rather than named, so a
    # third run needs no code change to be included.
    retests = []
    root = settings.raw_dir / "retest"
    for d in sorted(root.iterdir()) if root.exists() else []:
        if d.is_dir():
            scores = load_scores(d)
            if scores:
                retests.append((d.name, scores))
    if not retests:
        raise SystemExit("no retest passes under data/raw/retest/; run the "
                         "retest stage first")

    log.info("consolidating %d pass(es): %s", 1 + len(retests),
             ", ".join(["base"] + [n for n, _ in retests]))
    rows = consolidate(list(first.values()),
                       *[list(s.values()) for _, s in retests])
    summary = consolidation_summary(rows)

    existing = read_table(settings.out_dir, "subtask_automation_scores")
    cols = tuple(existing[0].keys()) if existing else tuple(rows[0].keys())
    for extra in ("n_raters", "score_disagreement", "raters"):
        if extra not in cols:
            cols = cols + (extra,)
    COLUMNS["subtask_automation_scores"] = cols
    write_csv(settings.out_dir / "subtask_automation_scores.csv", rows, cols)

    # Rewriting the subtask table is not enough. The task- and
    # occupation-level tables are a PROPAGATION of it, written by the score
    # stage, and the susceptibility build reads all three - so leaving them
    # stale meant the consolidated scores changed nothing downstream and every
    # figure came out byte-identical, which is exactly what a silent no-op
    # looks like. Re-propagate from the consolidated subtask scores.
    from .score import propagate
    task_scores, occ_scores = propagate(
        rows,
        read_table(settings.out_dir, "tasks"),
        read_table(settings.out_dir, "task_subtasks"),
        read_table(settings.out_dir, "occupations"),
    )
    append_sqlite(settings.out_dir / "onet_stem.sqlite",
                  {"subtask_automation_scores": rows,
                   "task_automation_scores": task_scores,
                   "occupation_automation_scores": occ_scores})
    for name, table in (("task_automation_scores", task_scores),
                        ("occupation_automation_scores", occ_scores)):
        write_csv(settings.out_dir / f"{name}.csv", table, COLUMNS[name])
    log.info("re-propagated to %d tasks and %d occupations",
             len(task_scores), len(occ_scores))
    (settings.out_dir / "consolidation_report.json").write_text(
        json.dumps(summary, indent=2))

    # Record the real provenance where the display surfaces already look for it.
    # Three pages said "scored by claude-opus-5" after consolidation had made the
    # canonical scores a mean over three passes, and nothing could have caught
    # it because each was reading a field the score stage wrote and the
    # consolidate stage did not update.
    raters = sorted({r for row in rows for r in
                     str(row.get("raters", "")).split(";") if r and r != "?"})
    counts = {}
    for row in rows:
        for r in str(row.get("raters", "")).split(";"):
            if r and r != "?":
                counts[r] = counts.get(r, 0) + 1
    report_path = settings.out_dir / "scoring_report.json"
    if report_path.exists():
        meta = json.loads(report_path.read_text())
        meta["raters"] = raters
        meta["passes"] = max((row.get("n_raters") or 1) for row in rows)
        meta["rater_counts"] = counts
        meta["provenance"] = (
            f"the mean of {meta['passes']} independent scoring passes "
            f"({', '.join(raters)})" if meta["passes"] > 1 else raters[0]
            if raters else "a model")
        report_path.write_text(json.dumps(meta, indent=2))
        log.info("provenance recorded: %s", meta["provenance"])

    log.info("-" * 72)
    log.info("consolidated %d subtasks by rater count: %s",
             summary["subtasks"],
             ", ".join(f"{n} rater(s) x {c}"
                       for n, c in summary["by_rater_count"].items()))
    log.info("  disagreement: median %.1f · p90 %.1f · max %.1f points",
             summary["median_disagreement"] or 0, summary["p90_disagreement"] or 0,
             summary["max_disagreement"] or 0)
    log.info("  %d subtasks (%.1f%%) differ by more than 15 points and should "
             "carry a caveat wherever cited",
             summary["above_15_points"], 100 * (summary["share_above_15"] or 0))
    log.info("  re-run the report stage to propagate these into every index")
    return summary


def run_validate_doc(settings: Settings, *, fix: bool = False) -> dict[str, Any]:
    """Check METHODOLOGY.md's figures against the data they describe.

    With fix=True, regenerate the registered tables first. Only the tables whose
    every cell comes from a report file - the prose around them carries
    interpretation and no script should be rewriting that.
    """
    from .validate_doc import log_report, refresh, validate_doc

    md = Path(__file__).resolve().parents[1] / "METHODOLOGY.md"
    if not md.exists():
        raise SystemExit(f"{md} not found")
    if fix and refresh(md, settings.out_dir):
        log.info("regenerated the registered tables in %s", md.name)
    results = validate_doc(md, settings.out_dir)
    stale = log_report(results)
    (settings.out_dir / "validation_doc.json").write_text(json.dumps(results, indent=2))
    if stale:
        log.error("%d figure(s) in METHODOLOGY.md disagree with data/out. Either the "
                  "document is out of date or a stage has not been re-run.", stale)
        raise SystemExit(1)
    log.info("every registered figure is in step with the data")
    return {"results": results, "stale": stale}


def run_uncertainty(settings: Settings, *, trials: int = 400) -> dict[str, Any]:
    """Push the measured sampling noise back through the whole derivation."""
    from .uncertainty import bootstrap, coerce, noise_model, report
    from .score import load_scores

    base = load_scores(settings.raw_dir)
    same_model = None
    for d in sorted((settings.raw_dir / "retest").iterdir()) \
            if (settings.raw_dir / "retest").exists() else []:
        if not d.is_dir():
            continue
        scores = load_scores(d)
        if scores and next(iter(scores.values())).get("model") == \
                next(iter(base.values())).get("model"):
            same_model = scores
            break
    if not base or not same_model:
        raise SystemExit(
            "the noise model needs two passes of the SAME model. Two different "
            "models would overstate it - most of their disagreement is "
            "calibration, which a re-run does not resample. Run: retest "
            "--retest-model <the model that produced the first pass> "
            "--retest-label <something else>")

    noise = noise_model(list(base.values()), list(same_model.values()))
    log.info("sampling noise sd by dimension: %s",
             {k: round(v, 2) for k, v in noise.items()})

    scores = coerce(read_table(settings.out_dir, "subtask_automation_scores"))
    raw = bootstrap(scores,
                    read_table(settings.out_dir, "tasks"),
                    read_table(settings.out_dir, "task_subtasks"),
                    read_table(settings.out_dir, "occupations"),
                    noise, trials=trials)
    rep = report(raw,
                 read_table(settings.out_dir, "occupation_susceptibility"),
                 read_table(settings.out_dir, "occupation_handoff"),
                 json.loads((settings.out_dir / "scenarios_report.json").read_text())
                 if (settings.out_dir / "scenarios_report.json").exists() else {})

    per_occ = rep.pop("per_occupation")
    (settings.out_dir / "uncertainty_report.json").write_text(json.dumps(rep, indent=2))
    flat = [{
        "onet_soc_code": o["onet_soc_code"], "title": o["title"],
        **{f"susceptibility_{k}": o.get("susceptibility", {}).get(k)
           for k in ("mean", "sd", "p05", "p95")},
        "exposure_p05": o.get("exposure", {}).get("p05"),
        "exposure_p95": o.get("exposure", {}).get("p95"),
        "quadrant_published": o["quadrant"]["published"],
        "quadrant_holds": o["quadrant"]["holds"],
        "handoff_published": o["handoff"]["published"],
        "handoff_holds": o["handoff"]["holds"],
    } for o in per_occ]
    COLUMNS["occupation_uncertainty"] = tuple(flat[0].keys()) if flat else ()
    write_csv(settings.out_dir / "occupation_uncertainty.csv", flat,
              COLUMNS["occupation_uncertainty"])
    append_sqlite(settings.out_dir / "onet_stem.sqlite",
                  {"occupation_uncertainty": flat})

    log.info("-" * 72)
    log.info("%d resamples of %d occupations", rep["trials"], rep["occupations"])
    log.info("  median 90%% interval on susceptibility: %.1f points wide",
             rep["median_susceptibility_ci_width"] or 0)
    log.info("  quadrant label survives under half the time for %d occupations",
             rep["quadrant_unstable"])
    log.info("  handoff class survives under half the time for %d occupations",
             rep["handoff_unstable"])
    for name, s in rep["scenarios"].items():
        log.info("  %-12s automated %d, 90%% interval %d-%d (published %s)",
                 name, round(s["mean"]), round(s["p05"]), round(s["p95"]),
                 s["published"])
    return rep


def run_regress(settings: Settings) -> dict[str, Any]:
    """Run the multivariate fits on the already-built tables."""
    from .regress import COEFFICIENT_COLUMNS, build, log_report

    tables = {name: read_table(settings.out_dir, name) for name in (
        "occupation_susceptibility", "occupation_indices", "external_benchmarks",
        "soc_susceptibility", "task_susceptibility", "task_churn_occupations",
        "occupation_uncertainty")}
    if not tables["occupation_susceptibility"]:
        raise SystemExit("no occupation_susceptibility - run the report stage first")

    report = build(tables)
    if not report["fits"]:
        raise SystemExit("no specification had the tables it needs; see the "
                         "skipped list from earlier stages")

    rows = report["coefficient_rows"]
    COLUMNS["regressions"] = COEFFICIENT_COLUMNS
    write_csv(settings.out_dir / "regressions.csv", rows, COEFFICIENT_COLUMNS)
    append_sqlite(settings.out_dir / "onet_stem.sqlite", {"regressions": rows})
    (settings.out_dir / "regression_report.json").write_text(
        json.dumps({k: v for k, v in report.items() if k != "coefficient_rows"},
                   indent=2))

    log_report(report)
    return report
