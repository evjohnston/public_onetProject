"""Assemble docs/ for GitHub Pages.

Pages serves a directory, not a build step, so this copies the finished
artefacts out of data/out into docs/ and writes a landing page around them.
data/out stays the working directory the stages write to; docs/ is what the
world sees, refreshed explicitly rather than on every run.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DOMAIN = "onet.emersonjohnston.org"
REPO = "https://github.com/evjohnston/onet-project"

COPY = ("story.html", "dashboard.html", "security_matrix.html")
METHODOLOGY = Path("METHODOLOGY.md")
COPY_DIRS = ("figures",)
DATA_FILES = (
    "occupation_susceptibility.csv", "task_susceptibility.csv",
    "subtask_susceptibility.csv", "occupation_handoff.csv",
    "soc_susceptibility.csv", "external_benchmarks.csv",
    "transitions.csv", "stranded_occupations.csv", "wage_deciles.csv",
    "task_churn_occupations.csv", "task_churn_steps.csv", "tasks_wide.csv",
    "security_matrix.csv", "security_fields.csv",
)


def _read(out_dir: Path, name: str) -> dict[str, Any]:
    path = out_dir / name
    return json.loads(path.read_text()) if path.exists() else {}


def build_index(out_dir: Path) -> str:
    v = _read(out_dir, "validation_report.json").get("summary", {})
    emp = _read(out_dir, "employment_report.json").get("headline", {})
    ext = _read(out_dir, "external_validation.json")
    pw = _read(out_dir, "pathways_report.json")
    ch = _read(out_dir, "churn_report.json")
    sc = _read(out_dir, "scoring_report.json")
    unc = _read(out_dir, "uncertainty_report.json")
    rel = _read(out_dir, "reliability_report.json")

    human = next((c for c in ext.get("susceptibility_vs", [])
                  if c["measure"] == "human_gamma"), {})
    # Both of these were hardcoded literals and both had gone stale: Frey at
    # 0.006 against an actual 0.024, and a "9%" that corresponds to no quantity
    # the dataset computes under any plausible reading - the nearest candidates
    # are 30.7% (watch points), 20.3% (a wide willingness gap) and 45.8% (at a
    # weighty crossing). A figure in prose that nothing derives will rot, so
    # these are derived.
    frey = next((c for c in ext.get("susceptibility_vs", [])
                 if c["measure"] == "frey_osborne"), {})
    hand = _read(out_dir, "handoff_report.json")
    watch_share = hand.get("employment_share_at_watch_points")
    trans = pw.get("transitions", {})
    wages = pw.get("wages", {})

    stats = [
        (f"{v.get('occupations', 0)}", "STEM occupations"),
        (f"{v.get('tasks', 0):,}", "tasks scored"),
        (f"{v.get('distinct_subtasks', 0)}", "distinct subtasks"),
        (f"{emp.get('total_employment', 0)/1e6:.1f}M", "workers matched"),
        (f"{human.get('pearson', 0):.2f}", "vs human expert ratings"),
        (f"{trans.get('stranded', 0)}", "occupations with nowhere to move"),
    ]
    stat_html = "".join(
        f'<div class="stat"><div class="n">{n}</div><div class="k">{k}</div></div>'
        for n, k in stats)

    findings = [
        ("The old measure does not survive",
         f"Frey &amp; Osborne&rsquo;s 2013 computerisation scores correlate with LLM "
         f"exposure at <b>r&nbsp;=&nbsp;{frey.get('pearson', 0):.3f}</b> across the "
         f"{frey.get('n', 0)} occupations carrying both. Not weak agreement — none."),
        ("Capability is not what holds most work",
         f"Pay barely correlates with exposure (r&nbsp;=&nbsp;"
         f"{wages.get('wage_vs_exposure', 0)}) but does with accountability "
         f"(r&nbsp;=&nbsp;{wages.get('wage_vs_anchoring', 0)})."
         + (f" <b>{100*watch_share:.0f}%</b> of STEM workers are at a watch point: "
            f"capability is present and accountability is what holds the line."
            if watch_share is not None else "")),
        ("Exposure is clustered, so there is nowhere to go",
         f"<b>{trans.get('stranded', 0)} of "
         f"{trans.get('stranded', 0) + trans.get('with_a_destination', 0)}</b> "
         "occupations have no close, meaningfully safer neighbour in the activity "
         "network. Moving to an adjacent role fails where it is most needed."),
        ("And the work has already started moving",
         f"Across eleven years of archived releases, task turnover is "
         f"<b>{100*ch.get('refreshed_only', {}).get('turnover_rate', 0):.1f}%</b> "
         "among occupations O*NET actually re-surveyed — and roughly three times "
         "faster in the exposed ones than the protected ones."),
    ]
    find_html = "".join(
        f'<article><h3>{t}</h3><p>{b}</p></article>' for t, b in findings)

    data_links = "".join(
        f'<li><a href="data/{n}">{n}</a></li>' for n in DATA_FILES)

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#1c1c18">
<title>STEM work and the handoff to AI — an O*NET task dataset</title>
<meta name="description" content="287 STEM occupations, 5,717 tasks and 963 subtasks scored
for automation exposure, validated against human expert ratings at r = 0.85.">
<meta property="og:title" content="STEM work and the handoff to AI">
<meta property="og:description" content="Which tasks in which jobs are exposed to
automation — and what actually protects the rest.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=IBM+Plex+Mono:wght@400;500&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{{--ink:#1c1c18;--paper:#faf9f5;--paper2:#f2efe5;--paper3:#f8f6f0;
  --rust:#bf5540;--muted:#6e6e66;--line:rgba(28,28,24,.15);
  --sans:'Manrope',Helvetica,Arial,sans-serif;--serif:'DM Serif Display',Georgia,serif;
  --mono:'IBM Plex Mono',ui-monospace,monospace}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
  -webkit-font-smoothing:antialiased}}
a{{color:inherit}}
.wrap{{max-width:1080px;margin:0 auto;padding:0 clamp(20px,5vw,48px)}}
.hero{{background:var(--ink);color:var(--paper);padding:clamp(60px,11vh,130px) 0 clamp(46px,8vh,90px)}}
.eyebrow{{font:500 10.5px/1.6 var(--mono);letter-spacing:.14em;text-transform:uppercase;
  color:#d9a43a;margin:0 0 20px}}
h1{{font:400 clamp(2.1rem,5.6vw,4rem)/1.04 var(--serif);letter-spacing:-.022em;margin:0 0 22px;max-width:17ch}}
.dek{{font-size:clamp(1rem,1.5vw,1.14rem);line-height:1.65;color:#c9c6bc;max-width:60ch;margin:0}}
.cta{{display:flex;gap:12px;flex-wrap:wrap;margin-top:34px}}
.btn{{display:inline-block;padding:13px 22px;border-radius:3px;text-decoration:none;
  font:500 11px/1 var(--mono);letter-spacing:.11em;text-transform:uppercase}}
.btn.p{{background:#d9a43a;color:var(--ink)}}
.btn.ghost{{border:1px solid rgba(250,249,245,.32);color:var(--paper)}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:1px;background:var(--line);border-block:1px solid var(--line);margin:0}}
.stat{{background:var(--paper);padding:26px 22px}}
.stat .n{{font:400 clamp(1.7rem,3.4vw,2.3rem)/1 var(--serif);color:var(--rust);letter-spacing:-.02em}}
.stat .k{{font:500 9.5px/1.6 var(--mono);letter-spacing:.11em;text-transform:uppercase;
  color:var(--muted);margin-top:8px}}
section.body{{padding:clamp(52px,9vh,96px) 0}}
h2{{font:400 clamp(1.5rem,3vw,2.1rem)/1.15 var(--serif);letter-spacing:-.018em;margin:0 0 10px}}
.lead{{color:var(--muted);line-height:1.7;max-width:68ch;margin:0 0 40px}}
.finds{{display:grid;grid-template-columns:1fr 1fr;gap:28px 40px}}
@media(max-width:760px){{.finds{{grid-template-columns:1fr}}}}
.finds h3{{font:600 .95rem/1.4 var(--sans);margin:0 0 8px}}
.finds p{{margin:0;color:var(--muted);font-size:.92rem;line-height:1.68}}
.finds b{{color:var(--ink);font-weight:600}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px;margin-top:8px;align-items:stretch}}
@media(max-width:760px){{.cards{{grid-template-columns:1fr}}}}
/* Flex column so the call-to-action pins to the bottom of every card. As
   blocks, cards of unequal copy length left their links at ragged heights. */
.card{{display:flex;flex-direction:column;text-decoration:none;background:var(--paper3);
  border:1px solid var(--line);border-radius:4px;padding:26px 26px 28px;
  transition:transform .15s ease,box-shadow .15s ease}}
.card:hover{{transform:translateY(-2px);box-shadow:0 8px 26px rgba(28,28,24,.09)}}
.card .t{{font:400 1.35rem/1.2 var(--serif);margin:0 0 8px}}
.card .d{{color:var(--muted);font-size:.9rem;line-height:1.6;margin:0;flex:1 1 auto}}
.card .go{{font:500 10px/1 var(--mono);letter-spacing:.11em;text-transform:uppercase;
  color:var(--rust);margin:18px 0 0;display:block}}
.tint{{background:var(--paper2)}}
.gal{{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:14px}}
.gal a{{display:block;border:1px solid var(--line);border-radius:3px;overflow:hidden;background:var(--paper)}}
.gal img{{width:100%;height:150px;object-fit:cover;object-position:top left;display:block}}
.gal span{{display:block;padding:9px 12px;font:500 9.5px/1.5 var(--mono);
  letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}}
ul.data{{columns:2;gap:30px;padding-left:18px;margin:0}}
@media(max-width:600px){{ul.data{{columns:1}}}}
ul.data li{{font:500 11.5px/2.1 var(--mono);color:var(--muted)}}
footer{{border-top:1px solid var(--line);padding:38px 0 60px;color:var(--muted);font-size:.84rem;line-height:1.7}}
footer a{{color:var(--ink)}}
.note{{font-size:.84rem;color:var(--muted);line-height:1.7;max-width:70ch}}
</style></head>
<body>

<header class="hero"><div class="wrap">
  <p class="eyebrow">O*NET · {v.get('occupations', 0)} STEM occupations · {v.get('tasks', 0):,} tasks · {v.get('distinct_subtasks', 0)} subtasks</p>
  <h1>Which work is exposed to AI, and what protects the rest.</h1>
  <p class="dek">A scraped, scored and externally validated dataset of STEM work.
  Every task rated through the standardised activity beneath it, weighted by BLS
  employment, and checked against independent human expert ratings at
  r&nbsp;=&nbsp;{human.get('pearson', 0):.2f}.</p>
  <div class="cta">
    <a class="btn p" href="story.html">Read the story</a>
    <a class="btn ghost" href="dashboard.html">Open the dashboard</a>
    <a class="btn ghost" href="security_matrix.html">Security matrix</a>
    <a class="btn ghost" href="methodology.html">Methodology</a>
    <a class="btn ghost" href="{REPO}">Source &amp; data</a>
  </div>
</div></header>

<div class="stats">{stat_html}</div>

<section class="body"><div class="wrap">
  <h2>Four findings</h2>
  <p class="lead">Each comes from a different data source, and they point the same way.</p>
  <div class="finds">{find_html}</div>
</div></section>

<section class="body tint"><div class="wrap">
  <h2>Three ways in</h2>
  <p class="lead">The story walks through the argument in twelve chapters with two
  interactives. The dashboard lets you interrogate the same tables directly. The
  security matrix asks a different question of the same data: not whether a handoff
  will happen, but whether it could be undone.</p>
  <div class="cards">
    <a class="card" href="story.html">
      <p class="t">The story</p>
      <p class="d">Scroll-driven, twelve chapters. The inversion against Frey &amp;
      Osborne, what a job is made of, the handoff frontier, who actually does the work,
      and whether any of it has moved yet. Includes an explorer for any occupation and
      a side-by-side comparison of any two.</p>
      <span class="go">Read &rarr;</span>
    </a>
    <a class="card" href="dashboard.html">
      <p class="t">The dashboard</p>
      <p class="d">Every occupation, task and subtask, rankable by six different
      measures. Network projections, the employment join, the validation panel, and a
      PNG export on every chart.</p>
      <span class="go">Explore &rarr;</span>
    </a>
    <a class="card" href="security_matrix.html">
      <p class="t">The security matrix</p>
      <p class="d">Every occupation on three axes at once: what AI deployment buys,
      what it costs to remove the human, and how hard the capability would be to
      rebuild. Rotatable in 3D with axis-aligned face views for reading values, a
      scenario selector, and eight named cells from &ldquo;clear win&rdquo; to
      &ldquo;strategic trap&rdquo;.</p>
      <span class="go">Open &rarr;</span>
    </a>
  </div>
</div></section>

<section class="body"><div class="wrap">
  <h2>How it was built</h2>
  <p class="lead">Every source with its version and access date, every decision that
  shapes a number, and the ten things this dataset cannot tell you.</p>
  <div class="cards">
    <a class="card" href="methodology.html">
      <p class="t">Methodology</p>
      <p class="d">Sampling frame, extraction, the scoring rubric and its unit of
      analysis, the derived measures and why they are not simple means, three layers of
      validation, and the limitations stated plainly.</p>
      <span class="go">Read &rarr;</span>
    </a>
    <a class="card" href="{REPO}/blob/main/METHODOLOGY.md">
      <p class="t">On GitHub</p>
      <p class="d">The same document as markdown, alongside the pipeline that produced
      every figure in it, the 250 tests, and <code>manifest.json</code> with per-file row
      counts and content hashes for each run.</p>
      <span class="go">View source &rarr;</span>
    </a>
  </div>
</div></section>

<section class="body"><div class="wrap">
  <h2>Figures</h2>
  <p class="lead">Publication-resolution PNGs, rendered from the dashboard itself so they
  cannot drift from what it shows.</p>
  <div class="gal">__GALLERY__</div>
</div></section>

<section class="body tint"><div class="wrap">
  <h2>The data</h2>
  <p class="lead">Result tables as CSV. The full pipeline, tests and rebuild
  instructions are in the repository.</p>
  <ul class="data">{data_links}</ul>
</div></section>

<footer><div class="wrap">
  <p><b>Scores are model-generated.</b> The {v.get('distinct_subtasks', 0)} subtask ratings are
  {sc.get('provenance', sc.get('model', 'model-generated'))} under a versioned rubric, not a
  survey. They correlate with independent human expert ratings at
  r&nbsp;=&nbsp;{human.get('pearson', 0):.2f}, which is evidence of validity rather than a
  substitute for it. Two passes of the same model differ by
  {rel.get('test_retest_abs_diff', 3.8):.1f} points on average, and that noise carried through
  the whole derivation leaves a 90% interval of about
  {unc.get('median_susceptibility_ci_width', 1.8):.1f} points on an occupation's
  susceptibility &mdash; but {unc.get('quadrant_unstable', 0)} occupations have a quadrant label
  that survives fewer than half of those resamples and should not be cited at that level. See
  <a href="methodology.html#7-2-error-bars">&sect;7.2</a>.</p>
  <p>O*NET data is provided by the U.S. Department of Labor under
  <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>; O*NET® is a
  trademark of USDOL/ETA. Employment and wages from the BLS Occupational Employment and
  Wage Statistics (public domain). Benchmark ratings from Eloundou, Manning, Mishkin
  &amp; Rock (2023).</p>
  <p><a href="{REPO}">Source on GitHub</a></p>
</div></footer>

</body></html>"""


METHOD_PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Methodology \u2014 STEM work and the handoff to AI</title>
<meta name="description" content="Where the data comes from, how it was extracted and
scored, which decisions shape which numbers, and what the dataset cannot tell you.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=IBM+Plex+Mono:wght@400;500&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--ink:#1c1c18;--paper:#faf9f5;--paper2:#f2efe5;--paper3:#f8f6f0;
  --rust:#bf5540;--muted:#6e6e66;--line:rgba(28,28,24,.15);
  --sans:'Manrope',Helvetica,Arial,sans-serif;--serif:'DM Serif Display',Georgia,serif;
  --mono:'IBM Plex Mono',ui-monospace,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
  -webkit-font-smoothing:antialiased}
a{color:var(--rust)}
.top{background:var(--ink);color:var(--paper);padding:26px 0}
.top a{color:var(--paper);text-decoration:none;font:500 10.5px/1 var(--mono);
  letter-spacing:.12em;text-transform:uppercase}
.wrap{max-width:820px;margin:0 auto;padding:0 clamp(20px,5vw,40px)}
.lay{display:grid;grid-template-columns:230px minmax(0,1fr);gap:48px;
  max-width:1120px;margin:0 auto;padding:clamp(40px,7vh,72px) clamp(20px,5vw,40px) 90px}
@media(max-width:900px){.lay{grid-template-columns:1fr;gap:20px}}
nav.side{position:sticky;top:26px;align-self:start;font-size:.84rem;line-height:1.5}
@media(max-width:900px){nav.side{position:static}}
nav.side .h{font:500 9.5px/1.6 var(--mono);letter-spacing:.12em;text-transform:uppercase;
  color:var(--muted);margin-bottom:12px}
ol.toc{list-style:none;counter-reset:s;margin:0;padding:0}
ol.toc li{counter-increment:s;margin-bottom:9px}
ol.toc li::before{content:counter(s) ".";color:var(--muted);font:500 10px var(--mono);
  margin-right:7px;min-width:14px;display:inline-block}
ol.toc a{text-decoration:none;color:var(--ink)}
ol.toc a:hover{color:var(--rust)}
/* Subsections. Numbered from the heading itself because the prose refers to
   them as 7.1, not as "the fourth item under Validation". METHOD_PAGE is a
   plain string, not an f-string, so these braces are single: doubling them
   emitted literal {{ }} and silently invalidated every rule here, which showed
   up as the top-level counter running 3, 6, 11 because the sub-items were
   incrementing it. */
ol.toc ol.sub{list-style:none;margin:6px 0 10px;padding:0 0 0 2px}
ol.toc ol.sub li{counter-increment:none;margin-bottom:5px;font-size:.92em}
ol.toc ol.sub li::before{content:none}
ol.toc ol.sub a{color:var(--muted);display:flex;gap:8px;align-items:baseline}
ol.toc ol.sub a:hover{color:var(--rust)}
ol.toc ol.sub .n{font:500 9px var(--mono);opacity:.65;flex:0 0 1.8em}
article h1{font:400 clamp(2rem,4.4vw,2.9rem)/1.08 var(--serif);letter-spacing:-.022em;
  margin:0 0 10px}
article h2{font:400 clamp(1.3rem,2.4vw,1.72rem)/1.18 var(--serif);letter-spacing:-.016em;
  margin:52px 0 12px;padding-top:20px;border-top:1px solid var(--line)}
article h3{font:600 .96rem/1.4 var(--sans);margin:30px 0 8px}
article p{line-height:1.72;color:#333;margin:0 0 15px}
article li{line-height:1.7;color:#333;margin-bottom:7px}
article strong{color:var(--ink);font-weight:650}
article code{font:500 .86em var(--mono);background:var(--paper2);padding:1px 3px;margin:0 -1px;
  border-radius:3px}
article pre{background:var(--ink);color:#e9e6dd;padding:16px 18px;border-radius:4px;
  overflow:auto;font:500 12.5px/1.6 var(--mono)}
article pre code{background:none;padding:0;color:inherit}
article table{border-collapse:collapse;width:100%;margin:8px 0 22px;font-size:.87rem}
article td.num,article th.num{text-align:right;font-variant-numeric:tabular-nums;font-feature-settings:'tnum' 1}
article td.mid,article th.mid{text-align:center}
article th{text-align:left;font:500 9.5px/1.6 var(--mono);letter-spacing:.09em;
  text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--ink);
  padding:8px 10px 8px 0;vertical-align:bottom}
article td{padding:8px 10px 8px 0;border-bottom:1px solid var(--line);
  vertical-align:top;line-height:1.55}
article blockquote{margin:0 0 20px;padding:16px 20px;background:var(--paper3);
  border-left:2px solid var(--rust);border-radius:0 3px 3px 0}
article blockquote p{margin:0;font-size:.9rem;color:var(--muted)}
footer{border-top:1px solid var(--line);margin-top:56px;padding-top:24px;
  font-size:.82rem;color:var(--muted);line-height:1.7}
</style></head>
<body>
<div class="top"><div class="wrap"><a href="index.html">&larr; STEM work and the handoff to AI</a></div></div>
<div class="lay">
  <nav class="side"><div class="h">Contents</div>__TOC__</nav>
  <article>__BODY__
    <footer>__FOOTER__</footer>
  </article>
</div>
</body></html>"""


def publish(out_dir: Path, docs_dir: Path, domain: str = DOMAIN) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    for name in COPY:
        src = out_dir / name
        if src.exists():
            shutil.copy2(src, docs_dir / name)
        else:
            log.warning("%s missing - the site will link to a page that is not there", name)
    for d in COPY_DIRS:
        src = out_dir / d
        if src.exists():
            shutil.copytree(src, docs_dir / d, dirs_exist_ok=True)

    data_dir = docs_dir / "data"
    data_dir.mkdir(exist_ok=True)
    for name in DATA_FILES:
        src = out_dir / name
        if src.exists():
            shutil.copy2(src, data_dir / name)

    gallery = ""
    fig_dir = docs_dir / "figures"
    if fig_dir.exists():
        for png in sorted(p for p in fig_dir.glob("*.png") if "-dark" not in p.name):
            label = png.stem.split("-", 1)[-1].replace("-", " ")
            gallery += (f'<a href="figures/{png.name}"><img src="figures/{png.name}" '
                        f'alt="{label}" loading="lazy"><span>{label}</span></a>')

    if METHODOLOGY.exists():
        from .markdown import render, toc
        md = METHODOLOGY.read_text()
        (docs_dir / "methodology.html").write_text(
            METHOD_PAGE.replace("__TOC__", toc(md))
                       .replace("__BODY__", render(md))
                       .replace("__FOOTER__",
                                "Generated from METHODOLOGY.md. "
                                f'<a href="{REPO}/blob/main/METHODOLOGY.md">'
                                "View the source document</a>."),
            encoding="utf-8")
    else:
        log.warning("METHODOLOGY.md not found - the site will link to a missing page")

    (docs_dir / "index.html").write_text(
        build_index(out_dir).replace("__GALLERY__", gallery), encoding="utf-8")
    (docs_dir / "CNAME").write_text(domain + "\n")
    # Pages runs Jekyll by default, which skips files and folders beginning with
    # an underscore; this project has none today but the guard is free.
    (docs_dir / ".nojekyll").write_text("")

    size = sum(f.stat().st_size for f in docs_dir.rglob("*") if f.is_file())
    log.info("published %d files to %s (%.1f MB) for %s",
             sum(1 for f in docs_dir.rglob("*") if f.is_file()), docs_dir,
             size / 1e6, domain)
    return docs_dir
