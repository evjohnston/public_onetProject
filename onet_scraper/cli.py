"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from . import __version__
from .score import DEFAULT_MODEL as SCORE_DEFAULT_MODEL
from .build import build_all
from .config import STEM_CATEGORIES, TOP_LEVEL_CATEGORIES, Settings
from .http_client import PoliteClient
from .pipeline import (
    fetch_augment,
    fetch_bulk,
    fetch_occupations,
    fetch_stem_index,
    load_augment,
    load_bulk,
    load_failures,
    load_index,
    load_occupations,
    save_bulk_meta,
    save_index,
    write_manifest,
)
from .stages import (
    run_churn,
    run_employment,
    run_external,
    run_figures,
    run_network,
    run_pathways,
    run_publish,
    run_report,
    run_scenarios,
    run_security,
    run_validate_derived,
    run_validate_doc,
    run_smoke,
    run_retest,
    run_consolidate,
    run_uncertainty,
    run_scroller,
    run_score,
    run_regress,
)
from . import baseline
from .validate import log_report, validate

log = logging.getLogger("onet_scraper")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onet-scraper",
        description="Scrape O*NET OnLine STEM occupations into a job / task / subtask dataset.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="root for cache, raw snapshots and outputs (default: ./data)")
    parser.add_argument("--contact", default=None,
                        help="contact string for the User-Agent; also read from $ONET_CONTACT")
    parser.add_argument("--rate", type=float, default=1.5,
                        help="max requests per second across all workers (default: 1.5)")
    parser.add_argument("--workers", type=int, default=4, help="concurrent fetchers (default: 4)")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--cache-ttl-days", type=float, default=7.0,
                        help="re-fetch pages older than this; 0 disables expiry")
    parser.add_argument("--refresh", action="store_true",
                        help="ignore cache and checkpoints, re-fetch everything")
    parser.add_argument("--offline", action="store_true",
                        help="serve only from cache; fail on a cache miss")
    parser.add_argument("--ignore-robots", action="store_true",
                        help="skip the robots.txt check (the default paths are allowed)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N occupations (smoke tests)")
    parser.add_argument("--no-bulk", action="store_true",
                        help="skip the O*NET bulk crosswalk; task->subtask links will be empty")
    parser.add_argument("--with-descriptors", action="store_true",
                        help="fetch work context / abilities / activities / skills (~80 MB) "
                             "to build the collaboration and automation-bottleneck indices")
    parser.add_argument("--network-min-shared", type=int, default=3,
                        help="minimum shared subtasks for an occupation edge (default: 3)")
    parser.add_argument("--network-min-co-occurring", type=int, default=5,
                        help="minimum shared occupations for a subtask edge (default: 5)")
    parser.add_argument("--network-exclude-soc", default="",
                        help="comma-separated SOC prefixes to drop, e.g. '25' to remove "
                             "postsecondary teachers, whose O*NET profiles are near-identical")
    parser.add_argument("--score-model", default=SCORE_DEFAULT_MODEL,
                        help=f"model for the scoring stage (default: {SCORE_DEFAULT_MODEL})")
    parser.add_argument("--score-chunk-size", type=int, default=12,
                        help="subtasks rated per request (default: 12)")
    parser.add_argument("--score-workers", type=int, default=4)
    parser.add_argument("--domain", default=None,
                        help="custom domain for the publish stage (writes docs/CNAME)")
    parser.add_argument("--chrome", default=None,
                        help="path to a Chrome/Chromium binary for the figures stage")
    parser.add_argument("--dark", action="store_true",
                        help="for the figures stage: render the dark theme")
    parser.add_argument("--figure-scale", type=int, default=2,
                        help="device pixel ratio for exported PNGs (default: 2)")
    parser.add_argument("--only", default="",
                        help="for the figures stage: comma-separated name filters")
    parser.add_argument("--dry-run", action="store_true",
                        help="for the score stage: print the cost estimate and stop")
    parser.add_argument("--with-ratings", action="store_true",
                        help="also download task_ratings.csv, education.csv and "
                             "job_zones.csv (~30 MB) for task frequency, the "
                             "education distribution and bulk job zones")
    parser.add_argument("--save-html", action="store_true",
                        help="keep a readable copy of every occupation page under raw/html")
    parser.add_argument("--categories", default=None,
                        help="comma-separated top-level STEM page ids (default: all). "
                             f"Valid: {','.join(TOP_LEVEL_CATEGORIES)}. Sub-disciplines "
                             "are sections of these pages and are captured automatically.")
    parser.add_argument("--trials", type=int, default=400,
                        help="resamples for the uncertainty stage (default 400)")
    parser.add_argument("--fix-figures", action="store_true",
                        help="with the validate-doc stage, regenerate the "
                             "registered tables from the data before checking")
    parser.add_argument("--retest-label", default="",
                        help="name for this retest pass; defaults to the model "
                             "name. A second run of the same model needs its "
                             "own label or it would overwrite the first")
    parser.add_argument("--retest-model", default="claude-sonnet-5",
                        help="model for the retest stage. A different model from "
                             "the one that produced the first pass measures "
                             "cross-model agreement; the same model measures "
                             "test-retest reliability")
    parser.add_argument("--budget", type=float, default=4.0,
                        help="hard cost ceiling for the retest stage in USD "
                             "(default 4.00); the run refuses to start above it")
    parser.add_argument("--accept-baseline", action="store_true",
                        help="adopt this run's row counts even where they moved "
                             "sharply from the previous run (use when a new "
                             "O*NET release genuinely changed them)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")

    parser.add_argument(
        "stage",
        nargs="?",
        default="run",
        choices=["run", "fetch-stem", "fetch-occupations", "fetch-bulk",
                 "fetch-descriptors", "build", "validate", "network", "score",
                 "report", "employment", "validate-external", "figures", "story", "churn", "pathways", "publish", "scenarios", "security", "validate-derived", "validate-doc", "smoke", "retest", "consolidate", "uncertainty", "regress",
                 "clean-cache"],
        help="which stage to run (default: run = all of them)",
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    categories = TOP_LEVEL_CATEGORIES
    if args.categories:
        requested = tuple(c.strip() for c in args.categories.split(",") if c.strip())
        unknown = [c for c in requested if c not in TOP_LEVEL_CATEGORIES]
        if unknown:
            raise SystemExit(f"unknown STEM category id(s): {', '.join(unknown)}")
        categories = requested
    return Settings(
        data_dir=args.data_dir,
        contact=args.contact,
        rate=args.rate,
        workers=max(1, args.workers),
        timeout=args.timeout,
        max_retries=args.max_retries,
        cache_ttl_days=args.cache_ttl_days,
        refresh=args.refresh,
        offline=args.offline,
        obey_robots=not args.ignore_robots,
        limit=args.limit,
        use_bulk=not args.no_bulk,
        with_ratings=args.with_ratings,
        with_descriptors=args.with_descriptors,
        save_html=args.save_html,
        categories=categories,
    )


def make_client(settings: Settings) -> PoliteClient:
    return PoliteClient(
        settings.cache_dir,
        settings.user_agent,
        rate=settings.rate,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
        cache_ttl_days=settings.cache_ttl_days,
        refresh=settings.refresh,
        offline=settings.offline,
        obey_robots=settings.obey_robots,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    settings = settings_from_args(args)
    settings.ensure_dirs()
    stage = args.stage

    if stage == "clean-cache":
        for target in (settings.cache_dir, settings.raw_dir):
            if target.exists():
                shutil.rmtree(target)
                log.info("removed %s", target)
        return 0

    if stage == "network":
        exclude = tuple(p.strip() for p in args.network_exclude_soc.split(",") if p.strip())
        run_network(settings, min_shared=args.network_min_shared,
                    min_co_occurring=args.network_min_co_occurring, exclude_soc=exclude)
        return 0

    if stage == "figures":
        run_figures(settings, chrome=args.chrome, dark=args.dark,
                    scale=args.figure_scale,
                    only=tuple(o.strip() for o in args.only.split(",") if o.strip()))
        return 0

    if stage == "story":
        run_scroller(settings)
        return 0

    if stage == "publish":
        run_publish(settings, args.domain)
        return 0

    if stage == "scenarios":
        run_scenarios(settings)
        return 0

    if stage == "security":
        run_security(settings)
        return 0

    if stage == "validate-derived":
        run_validate_derived(settings)
        return 0

    if stage == "validate-doc":
        run_validate_doc(settings, fix=args.fix_figures)
        return 0

    if stage == "smoke":
        run_smoke(settings, chrome=args.chrome)
        return 0

    if stage == "uncertainty":
        run_uncertainty(settings, trials=args.trials)
        return 0

    if stage == "consolidate":
        run_consolidate(settings)
        return 0

    if stage == "retest":
        report = run_retest(settings, model=args.retest_model,
                            chunk_size=(args.score_chunk_size
                                        if "--score-chunk-size" in sys.argv else 0),
                            workers=args.score_workers, budget_usd=args.budget,
                            label=args.retest_label)
        return 1 if report.get("failures") else 0

    if stage == "pathways":
        run_pathways(settings)
        return 0

    if stage == "report":
        run_report(settings)
        return 0

    if stage == "score":
        report = run_score(settings, model=args.score_model,
                           chunk_size=args.score_chunk_size, workers=args.score_workers,
                           refresh=args.refresh, dry_run=args.dry_run)
        return 1 if report.get("failures") else 0

    client = make_client(settings)

    if stage == "regress":
        run_regress(settings)
        return 0

    if stage == "churn":
        run_churn(settings, client)
        return 0

    if stage == "employment":
        run_employment(settings, client)
        return 0

    if stage == "validate-external":
        run_external(settings, client)
        return 0

    if stage in ("run", "fetch-stem"):
        log.info("stage 1: STEM roster")
        index = fetch_stem_index(client, settings)
        save_index(settings, index)
        log.info("roster: %d STEM occupations, %d category memberships",
                 len(index["occupations"]), len(index["memberships"]))
        if stage == "fetch-stem":
            return 0
    else:
        index = load_index(settings)

    if settings.limit:
        # Trim the roster itself so every downstream table and check sees the same
        # universe; otherwise --limit looks like 280 failed fetches.
        kept = {o["onet_soc_code"] for o in index["occupations"][: settings.limit]}
        index["occupations"] = [o for o in index["occupations"] if o["onet_soc_code"] in kept]
        index["memberships"] = [m for m in index["memberships"] if m[0] in kept]
        log.info("--limit %d: roster trimmed to %d occupations", settings.limit, len(kept))

    if stage in ("run", "fetch-occupations"):
        log.info("stage 2: occupation detail reports")
        codes = [o["onet_soc_code"] for o in index["occupations"]]
        records, failures = fetch_occupations(client, settings, codes)
        log.info("parsed %d occupations (%d failed)", len(records), len(failures))
        if stage == "fetch-occupations":
            return 1 if failures else 0
    else:
        records = load_occupations(settings, [o["onet_soc_code"] for o in index["occupations"]])
        failures = load_failures(settings)

    bulk = None
    if settings.use_bulk:
        if stage in ("run", "fetch-bulk"):
            log.info("stage 3: O*NET bulk task->subtask crosswalk")
            bulk = fetch_bulk(client, settings)
            save_bulk_meta(settings, bulk)
            if stage == "fetch-bulk":
                return 0
        else:
            bulk = load_bulk(settings)
            if bulk is None:
                log.warning("no bulk crosswalk on disk; run the fetch-bulk stage for "
                            "task-level subtasks")

    augment = None
    if settings.with_descriptors or stage == "fetch-descriptors":
        if stage in ("run", "fetch-descriptors"):
            log.info("stage 3b: O*NET descriptor files")
            augment = fetch_augment(client, settings)
            if stage == "fetch-descriptors":
                return 0
        else:
            augment = load_augment(settings)
    elif stage not in ("fetch-bulk",):
        augment = load_augment(settings)

    if stage == "fetch-bulk":
        return 0

    log.info("stage 4: building dataset in %s", settings.out_dir)
    tables = build_all(settings, index, records, bulk, augment)

    checks, summary = validate(tables, index, records, failures, settings.out_dir)
    errors = log_report(checks, summary)

    # Compare row counts with the previous run before the manifest is
    # overwritten. The Green-Task-Statements swap in release 24.0 read a
    # 140-occupation subset instead of the full file and nothing noticed,
    # because the manifest recorded the counts and nobody read them back.
    previous = baseline.load(settings.out_dir)
    drift = baseline.compare(baseline.counts_of(previous),
                             baseline.counts_of({
                                 "row_counts": {k: len(v) for k, v in tables.items()},
                                 "bulk_provenance": bulk["provenance"] if bulk else {},
                                 "descriptor_provenance": (
                                     augment["provenance"] if augment else {}),
                             }))
    drift_errors = baseline.log_report(drift, bool(previous))
    if drift_errors and not args.accept_baseline:
        errors += drift_errors

    write_manifest(settings, {
        "tool": f"onet-job-taskings {__version__}",
        "stem_source": "https://www.onetonline.org/find/stem?t=0",
        "bulk_release": bulk["release"] if bulk else None,
        "descriptor_provenance": augment["provenance"] if augment else {},
        "bulk_provenance": bulk["provenance"] if bulk else {},
        "settings": {
            "rate": settings.rate, "workers": settings.workers,
            "categories": list(settings.categories), "use_bulk": settings.use_bulk,
            "with_ratings": settings.with_ratings, "limit": settings.limit,
            "with_descriptors": settings.with_descriptors,
        },
        "http_stats": client.stats,
        "row_counts": {name: len(rows) for name, rows in tables.items()},
        "validation_summary": summary,
        "license": "O*NET data is provided by the U.S. Department of Labor under "
                   "CC BY 4.0. Attribute O*NET when redistributing.",
    })

    if errors:
        log.error("%d validation error(s); see %s", errors,
                  settings.out_dir / "validation_report.json")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
