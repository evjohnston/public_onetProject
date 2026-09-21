# Methodology

How this dataset was built, from which sources, with which decisions and which
known weaknesses. Written so the numbers can be checked rather than taken.

Every figure quoted below is reproduced in `data/out/*.json` by the run that
produced it; the commands to rebuild are in the README.

## 1. What the dataset is

A record of **which tasks belong to which STEM occupations, which standardised
activities those tasks are built from, and how exposed each of those is to being
performed by AI** — joined to employment and wages, and benchmarked against
independently produced measures.

| | Count |
| --- | --- |
| STEM occupations | 287 |
| Task statements | 5,717 |
| Distinct detailed work activities ("subtasks") | 963 |
| Task → activity links | 7,215 |
| Occupations with automation scores | 268 |
| Tasks with automation scores | 5,612 |
| Workers matched via BLS | 21,523,100 |

The three counts that differ from their headline are not losses to be explained
away: 19 occupations are SOC "All Other" residuals with no task data anywhere in
O\*NET, and 105 tasks postdate the activity crosswalk (§3.2).

## 2. Sampling frame

The universe is **every occupation O\*NET classifies as STEM**, taken from its
own STEM filter rather than from a keyword rule or an SOC-range heuristic. This
makes the boundary O\*NET's editorial decision, which is citable, rather than
ours, which would not be.

The filter has six top-level pages (`?t=0` through `?t=5`). Two of them split
into sub-disciplines rendered as anchored sections *within* the same page
(`?t=1#f1` … `#f4`), not as separate URLs — a fact that has to be handled or two
thirds of the category memberships are silently lost.

Membership is **many-to-many**: the individual category pages hold 302 rows
against 287 unique occupations, because 15 occupations appear under more than one
STEM discipline. Analyses should use `occupation_stem_categories` rather than
assuming one category per occupation.

## 3. Sources

| Source | What it supplies | Version | Accessed | Terms |
| --- | --- | --- | --- | --- |
| [O\*NET OnLine](https://www.onetonline.org/find/stem?t=0) | STEM roster, task statements, importance ratings, occupation-level activities | live site | 2026-09-16 20:53–20:58 UTC | CC BY 4.0 |
| [O\*NET Database](https://www.onetcenter.org/database.html) | task → activity crosswalk, activity hierarchy, descriptors | **31.0** | 2026-09-16 | CC BY 4.0 |
| O\*NET Database archive | archived task statements for the churn series | 20.1, 22.0, 24.0, 26.0, 28.0, 29.0, 30.0, 31.0 | 2026-09-16 | CC BY 4.0 |
| [BLS OEWS](https://www.bls.gov/oes/tables.htm) | employment and wages, national cross-industry | **May 2025** | 2026-09-16 | public domain |
| [Eloundou et al. (2023)](https://github.com/openai/GPTs-are-GPTs) | human and GPT-4 exposure ratings; Frey & Osborne, Felten/Raj/Seamans, Brynjolfsson SML | repo `main` | 2026-09-16 | see repo |
| Claude Opus 5 | subtask automation scores | rubric `2026-09-16.1` | 2026-09-16 | this repo, MIT |

Row counts and SHA-256 digests for every file fetched are recorded per run in
`data/out/manifest.json`.

### 3.1 Why the site *and* the bulk database

They are not redundant. The live site carries the STEM classification and the
current task ratings; the bulk database carries the crosswalk the site does not
publish. Both are needed and each is used only for what it alone provides.

### 3.2 The crosswalk the website does not expose

O\*NET OnLine publishes an occupation's tasks and, separately, its detailed work
activities. It does **not** publish which activities a given task maps to — the
`/link/moreinfo/task/<id>` endpoint returns related occupations, not activities.
That edge exists only in the bulk `tasks_to_dwas.csv`, so it is taken from there
and joined on the O\*NET task id, which the site exposes in each task's popup
link.

Consequence: the crosswalk is fixed at release 31.0 while the site is live, so
**105 of 5,717 tasks have no activity mapping** — they were added to the site
after the release was cut. That is a ceiling imposed by O\*NET's publication
cycle, not a scraping gap, and it is reported as such (98.2% coverage).

## 4. Extraction

### 4.1 Web scraping

287 occupation detail reports plus 6 category pages, fetched at 1.5 requests per
second across 4 workers with jitter, an identifying User-Agent, exponential
backoff honouring `Retry-After`, and `robots.txt` respected. The paths used are
permitted. Every response is cached on disk by URL digest with its fetch time and
content hash, so a rebuild needs no network at all.

Parsing is driven by O\*NET's own `data-title` cell labels and `data-text`
sort values rather than column position, so a re-ordered or re-styled table
cannot silently scramble a column.

**Two layouts exist.** Occupations with incumbent survey ratings render tasks as a
table; occupations without render the same section as a `<ul>` list with no
importance or category. Both are parsed. Missing this costs six occupations
their entire task list.

**Self-check.** O\*NET prints the row count of each section ("… 16 displayed").
The build compares that figure against the rows it actually parsed and fails if
they disagree. This is the check that catches a truncated table, and it is the
only reason the list-layout omission was found.

**Unrated scores are missing, not zero.** Tasks O\*NET has not rated display
"Not available" behind a `data-text="-2"` sort sentinel. Parsed naively that
sentinel enters the data as a score of −2 and drags every mean down. 264 of 5,717
tasks have no importance score and the field is left empty: 157 `New`, 102 from the
six occupations with no incumbent survey, and 5 `Supplemental` tasks O\*NET left
unrated. A validation check rejects any rating
outside 0–100.

### 4.2 Employment

The OEWS national cross-industry file, latest release discovered at runtime from
the tables page. Only `O_GROUP = "detailed"` rows are used. BLS suppression
markers (`*`, `**`, `#`, `~`) are parsed as missing rather than as numbers.

### 4.3 Archived releases

Eight releases spanning 2015–2026, taken as `db_<release>_text.zip`. Availability
is verified per release rather than assumed: `db_20_0` does not exist (that series
begins at 20.1) and an unpublished release is dropped from the span with a warning
instead of failing the run.

### 4.4 Optional measurement files

Three files that are free but large, fetched by `fetch-bulk --with-ratings`.
Every measure below degrades to what it did before if they are absent, so the
pipeline runs without them.

| File | Scale | What it gives |
| --- | --- | --- |
| `task_ratings.csv` | FT, 7 bands | frequency of each task, i.e. recurrence |
| `education.csv` | RL, 12 categories | required level of education, as a distribution |
| `job_zones.csv` | 1–5 | job zone at bulk grain rather than scraped |

**Recurrence** is the FT distribution collapsed to one number per task. The bands
are named, not numbered — "yearly or less" through "hourly or more" — so each is
mapped to an approximate rate per year and the **logs** are averaged. Averaging
the band numbers instead would treat the step from monthly to weekly as equal to
the step from daily to hourly, which is wrong by a factor of twenty in the
quantity being measured. The per-year rates are ours, not O\*NET's, and a test
asserts the ranking does not depend on them.

Recurrence is **measured and reported but deliberately not folded into
tractability** (§6.2). Against the three terms already on that axis it
correlates at **r = −0.53**, and the components say why:

| | r with recurrence |
| --- | ---: |
| `llm_exposure` | −0.53 |
| `physical_embodiment_required` | +0.53 |
| `judgment_under_uncertainty` | −0.04 |

The most repetitive work in this corpus is the most physically embodied and the
least exposed to language models. Emergency medicine physicians, physician
assistants and orthodontists score highest because they repeat procedures;
anthropologists and nuclear engineers score lowest because their work is rare.
That is a result worth stating on its own, and it is also the reason not to
average the term in: at r = −0.53 it cancels much of the existing signal, the
axis spread falls by a third, and because `TRACTABILITY_FLOOR` and `FRONTIER_K`
were calibrated against the wider distribution, 24 occupations shift out of
"human held" and 24 into "handed off" — which would read as AI having taken over
a quarter more of the corpus when nothing about the world had changed. Using it
properly requires re-deriving the frontier, which is a judgment, so it is left
undone and visible.

**Measurement precision.** Every descriptor table in this dataset carries `n` and
`standard_error` and nothing read them: an importance rating from 4 incumbents
was treated exactly like one from 225. `onet_ratings.rating_precision` now
reports both. The result is reassuring rather than alarming — the median STEM
occupation rests on 27 respondents and **none** falls below ten — but that was
worth establishing rather than assuming.

## 5. Scoring

### 5.1 Unit of analysis

Scores are assigned to the **963 distinct activities**, not the 5,717 tasks.

This is a substantive choice, not an economy. Rating the activity layer is six
times less work, but more importantly it removes an artefact: the same activity
cannot receive one score inside nursing and a different score inside engineering.
Consistency comes from the structure rather than from the rater's discipline.
Scores then propagate outward (§5.4).

### 5.2 Rubric

Seven dimensions, each 0–100, defined in `onet_scraper/score.py` and stamped on
every row as `rubric_version`:

| Dimension | Asks |
| --- | --- |
| `automation_feasibility_today` | could *deployed* technology do this end-to-end now |
| `llm_exposure` | could a current model do the cognitive core, given tools |
| `physical_embodiment_required` | does it need hands on matter in situ |
| `interpersonal_demand` | is the human relationship the substance of the work |
| `judgment_under_uncertainty` | could reasonable experts disagree |
| `accountability_requirement` | must an identifiable human answer for it |
| `error_cost` | how severe and irreversible is a mistake |

Plus a four-way verdict (`largely_automatable` / `augmentable` / `resistant` /
`human_anchored`), a confidence level, and a one-sentence rationale naming the
binding constraint.

The dimensions are deliberately separable. An activity can score high on
`llm_exposure` **and** be `human_anchored` — drafting a legal opinion is largely
language work that a named human must still own. 92 of the 963 activities score
≥70 on both. Collapsing these into a single "automation risk" number is the
specific failure the rubric is built to avoid.

### 5.3 Model and parameters

Claude Opus 5, adaptive thinking, structured output validated against a schema,
12 activities per request, 81 requests, **0 failures**, ≈ $8.20. Rubric
`2026-09-16.1`, fingerprint `49b87187f6f94ad8`. Results are cached per activity
keyed on the rubric version, so changing the rubric re-runs rather than mixing
incomparable scores.

### 5.4 Propagation

- **activity → task**: unweighted mean over the activities the task maps to
- **task → occupation**: mean weighted by O\*NET task **importance**, so an
  occupation is characterised by what matters in it rather than by its longest tail

Tasks with no scored activity are absent, not zero.

## 6. Derived measures

### 6.1 Susceptibility index

Not a mean of the seven dimensions. Two pairs are close to collinear in the
scored data:

| Pair | r |
| --- | --- |
| `llm_exposure` ↔ `physical_embodiment_required` | −0.89 |
| `accountability_requirement` ↔ `error_cost` | +0.90 |

Averaging all seven would double-count "can a machine do it" and double-count
"how much is at stake". Each pair collapses to one factor first, leaving:

- **exposure** = `llm_exposure` (physical embodiment is its mirror, not added again)
- **anchoring** = mean of stakes, interpersonal demand, judgment, physical embodiment

`susceptibility = 50 + (exposure − anchoring) / 2`, clamped to 0–100.

### 6.2 Handoff framework

`occupation_handoff.csv` maps the same dimensions onto the framework in Watson
(2026), which scores work on **tractability** (can AI lead it) and **resistance**
(will it be permitted to), and locates each unit on a six-stage scale of cognitive
leadership. Stage is the **lower** of the capability ceiling and the permission
ceiling, not an average — Watson's first four properties decide what AI can do,
the last two what it is allowed to do.

Six of his eight properties are covered. **Recurrence** is now measured (§4.4)
but deliberately not folded into tractability; **feedback speed and clarity** is
not in O\*NET at all and would need new scoring.

**The frontier is calibrated, and the calibration now travels with the
distribution.** Watson's figure shows the curve's shape and no numbers, so the
thresholds were chosen by eye against release 31.0: `FRONTIER_K = 53`,
`TRACTABILITY_FLOOR = 50`, `CROSSING_BAND = 5`, plus three literals inside the
watch-point rule. That made them quietly fragile. A threshold on an axis is
meaningful only relative to the spread of that axis, and the axis is a mean of
terms that can be added to or reweighted — so any change to the rubric would
silently reclassify the corpus.

That is not hypothetical. Averaging recurrence into tractability narrows the
axis by a third (sd 9.95 → 6.44), because recurrence runs against the other
terms at r = −0.53. Under the fixed thresholds the result would have been
reported as a large move toward automation. Under thresholds that track the
axis, it moves the other way:

| Calibration | Human held | Watch point | Crossing now | Handed off |
| --- | ---: | ---: | ---: | ---: |
| 3-term, adaptive *(published)* | 105 | 29 | 98 | 36 |
| 4-term, stale absolutes | 81 | 34 | 93 | **60** |
| 4-term, adaptive | 104 | 34 | 105 | **25** |

The stale constants did not merely exaggerate the effect, they **inverted its
direction**: "handed off" rises from 36 to 60 under fixed thresholds and falls
to 25 once the calibration adapts. Eighty-four occupations change class the
first way, fifty-four the second.

Each constant is therefore expressed as a **quantile of the observed
distribution** — the tractability floor at the 19.1st percentile, the frontier
at the 38.6th percentile of the *product* `T × R` (which is what the curve is a
threshold on), the crossing band at 0.391 standard deviations of resistance. The
quantiles were obtained by inverting the original absolutes against release 31.0,
so on that release they reproduce the old thresholds to four decimal places and
**not one classification changes**. A test asserts that identity. The
reparameterisation is not meant to move today's answer; it is meant to ensure
tomorrow's answer moves for a reason.

One detail worth recording: the precision of the crossing-band coefficient is
load-bearing. Architects sit at a frontier margin of +4.999362, within 0.0007 of
the band edge, so rounding 0.39117889 to 0.391 moves them from "crossing now" to
"human held". Their classification was never really determined by the data, and
the same is presumably true of any occupation near the curve.

### 6.3 Employment weighting

O\*NET reports at the 8-digit O\*NET-SOC level; OEWS reports employment at the
6-digit SOC level. **37 SOC codes here contain more than one O\*NET occupation.**
Attaching OEWS employment to each O\*NET row counts the same workers repeatedly —
3.38 million registered nurses five times over.

So susceptibility is aggregated to the SOC level **first**, then employment is
attached exactly once. `susceptibility_sd_within_soc` reports the spread across
each SOC's constituent occupations, so a mean concealing real disagreement is
visible. Where BLS publishes only a broad group (29-2011 and 29-2012 both appear
as 29-2010) codes fall back to it — and because both then land in the same group,
the fallback cannot double-count either.

### 6.4 Wage deciles

Employment-weighted, and an occupation's workers are **split across** bucket
boundaries rather than assigned whole. Registered nurses alone are 16% of these
workers — larger than a decile — so whole assignment produced buckets ranging
from 0.4M to 3.8M and the word "decile" was not true. Each decile now holds
2,152,310 workers.

Occupation-level correlations and employment-weighted deciles answer different
questions and point different ways here. Both are reported; quoting only one is a
choice about which question to answer.

### 6.5 Transition pathways

A destination must clear three bars: sufficient activity **overlap** (cosine
≥ 0.12), meaningful **relief** (≥ 12 susceptibility points), and **direction** —
the shared activities must include the destination's protected work, or the mover
carries their exposure with them.

All three thresholds are arbitrary, so the stage sweeps them. The stranded count
ranges **105 to 240** of 268 across the sweep: it is sensitive to how much relief
is demanded and nearly insensitive to the overlap floor. **The clustering of
exposure is the finding; the number 161 is not.**

### 6.6 Scenarios

`task_fates.csv` classifies every task as **automated**, **augmented** or
**unchanged** under three assumption sets. These carry **no dates and are not
forecasts.**

The mechanism is deliberately the variable Watson identifies as the real one.
Capability is roughly fixed and already measured; what differs between scenarios
is willingness to hand over accountability.

The cuts are **quantiles of the corpus's own distribution**, not absolute points
on the scale — see §7.1 for the measurement that forced that change, and the
`REFERENCE_QUANTILES` constants in `scenarios.py` for the values. On this corpus
they resolve to roughly exposure 75, 65 and 54:

| Scenario | Automates when | Automated tasks |
| --- | --- | ---: |
| Modest | 68th pct exposure, anchoring under the 32nd | 1,166 (21%) |
| Substantial | 47th pct exposure, anchoring under the 68th | 2,621 (47%) |
| Extreme | 31st pct exposure, anchoring under the 94th | 3,860 (69%) |

Tasks below the augment threshold are unchanged at every setting — for a nurse
midwife, 7 of 21 tasks never move regardless of scenario, which is the point the
chapter makes.

The fourth category, **new tasks, is not modelled**. It comes from O\*NET's own
`emerging_tasks` file — 121 statements flagged as new or revised work in these
occupations — so the arrival of new work is observed rather than assumed.

Employment flows are summed **per SOC code, once**. The first implementation
summed across O\*NET occupations and reported 49.3 M workers against a true
21.5 M, because 37 SOC codes hold several O\*NET occupations that each carry the
same employment figure. The O\*NET rows are collapsed to their SOC first; a test
covers it.

### 6.7 Task churn

Task statements diffed across the eight archived releases. A task **survives** if
its id persists **or** its normalised text matches one in the later release for
the same occupation — ids are retired and reissued, and a reworded task is not a
new one.

The 2019 SOC revision renumbered codes, so **93 of 268 occupations cannot be
compared** across the span at all. Those are reported as skipped, never counted
as retired tasks.

**The confound, stated plainly.** O\*NET re-surveys occupations on a rolling
cycle, so an occupation whose tasks did not change may simply not have been
looked at. Splitting on the `Date` column separates the two: 8.1% turnover among
the 153 occupations re-surveyed since 2022, 2.4% among the 115 that were not. The
unsplit 5.2% figure is diluted by occupations nobody checked, and no care with the
diff fixes that.

### 6.8 National security matrix

`security_matrix.csv` places every occupation on three axes at once. The first
two ask whether a handoff is *attractive*; the third asks whether it is
*reversible*, which is the question the other measures in this dataset do not
reach.

| Axis | Built from |
| --- | --- |
| **x** AI efficiency | importance-weighted share of the occupation's task mass AI can carry — automated tasks at full credit, augmented at 0.5 |
| **y** Risk of removing humans | 0.45 × error cost + 0.35 × accountability + 0.20 × judgment under uncertainty, importance-weighted |
| **z** Reconstitution difficulty | 0.50 × training depth + 0.30 × workforce scarcity + 0.20 × isolation |

**Unrated tasks still carry weight.** O\*NET does not rate task importance for
every occupation: 159 of 5,612 tasks here carry none, and for six occupations —
Cardiologists, Pediatric and Orthopedic Surgeons, Emergency Medical Technicians,
Hydrologic Technicians, Health Information Technologists — *every* task is
unrated. The first implementation skipped unrated tasks, which left those six
with a zero denominator, returned 0.0 on both axes, and filed surgeons under
"nothing is pushing this work toward AI". An unrated task now takes the mean
importance of the rated tasks in the same occupation, and an occupation with
nothing rated falls back to equal weighting — which says we do not know which of
its tasks matters more, not that none of them matters. Radiologists, who had 13
of 30 tasks unrated, move cell as a result, so this is not only about the six.

Efficiency is deliberately **not** raw exposure. An occupation whose one exposed
task carries 3% of its importance mass is not an efficiency opportunity, and
averaging exposure across tasks would score it as though it were. Because
efficiency reads the per-task verdicts from §6.6, it moves with the scenario:
mean 42.6 at modest, 58.9 at substantial, 73.8 at extreme.

**Why the risk weights exclude two dimensions we already have.** This index does
not reuse `anchoring`, which averages stakes, interpersonal demand, judgment and
physical embodiment. Two of those do not belong in a security reading:
interpersonal demand is a service-quality property — a call centre scores high
and carries no strategic risk — and physical embodiment is a capability *limit*,
not a consequence, so work a robot cannot reach is not thereby high-stakes.
Including them produces a general "human-centred work" index; excluding them is
what makes this a risk index. That single choice is the main thing to argue with
here, and the weights themselves are a calibrated judgment, not a measurement.

Training depth comes from the **Required Level of Education** distribution
(§4.4), averaged as years of schooling and rescaled so 10 years is 0 and
post-doctoral training is 100. It yields 191 distinct values across 268
occupations. **Job Zone** is the fallback for the 44 occupations O\*NET has not
surveyed for education, and `depth_source` on every row records which was used. Scarcity is log-scaled employment — the
corpus spans four orders of magnitude, and a linear read would call everything
except registered nurses scarce. Isolation is the inverse of `mean_similarity`
from the shared-activity network: where many neighbouring occupations share the
work, people can cross-train in, and reconstitution is easier.

The cube is split at 50 on each axis into eight named cells:

| Cell | Modest | Substantial | Extreme |
| --- | ---: | ---: | ---: |
| Strategic trap | 15 / 2.0% | 63 / 12.8% | 97 / 19.7% |
| Guard the pipeline | 59 / 8.3% | 69 / 10.0% | 79 / 11.2% |
| Protect | 99 / 22.3% | 51 / 11.5% | 17 / 4.5% |
| Reversible gamble | 6 / 5.3% | 18 / 12.9% | 26 / 24.3% |
| Quiet attrition | 22 / 3.0% | 12 / 1.3% | 2 / 0.1% |
| Clear win | 23 / 22.3% | 27 / 28.0% | 29 / 28.4% |
| Hold the line | 34 / 30.2% | 22 / 22.6% | 14 / 11.2% |
| Low stakes | 10 / 6.6% | 6 / 0.9% | 4 / 0.5% |

*(occupations / share of the 21,523,100 workers)*

**A correction, and why it is instructive.** An earlier version of this section
reported that no occupation anywhere in the corpus was a "reversible gamble", and
read that empty cell as a finding: the reconstitution axis had a floor of 44.3,
so nothing was genuinely easy to rebuild. That was an artefact of the
measurement, not a fact about the workforce. Training depth came from **Job
Zone**, which has five levels of which only three occur here, so it assigned the
same depth to an ophthalmic medical technician and a clinical neuropsychologist.
Replacing it with the **Required Level of Education** distribution (§4.4) — the
same question at twelve categories instead of five — drops the floor to 33.2 and
populates the cell with 21 occupations at the substantial scenario, 30 at the
extreme. A coarse input had manufactured a finding, and the finding was about
Job Zone.

The absolute split at 50 and the corpus median remain different questions, and
the matrix page still offers both: the first asks whether an occupation is
dangerous on an interpretable scale, the second which of these occupations is
most dangerous relative to the others.

**Employment partitions on a per-occupation share, not the SOC figure.** §6.3
collapses O\*NET occupations to SOC before summing, which is right for a total
and wrong for a partition: 18 of the 37 multi-occupation SOCs have members that
land in *different* cells. SOC 19-1029 splits four occupations across three cells,
and crediting each cell the SOC's full 55,850 makes the cell shares add to 130%.
Each SOC's employment is therefore divided evenly among its constituent
occupations. The even split is an assumption — O\*NET does not publish how a
SOC's workers divide — but every slice then sums exactly to 21,523,100 however
the reader cuts it. The alternative, classifying whole SOCs, would discard the
occupation-level distinctions the matrix exists to show.

Aggregated to field, weighting each axis by employment:

| Field | Workers | Eff | Risk | Recon | Cell |
| --- | --- | --- | --- | --- | --- |
| Healthcare Practitioners and Technical | 9,793,540 | 42.3 | 72.3 | 59.4 | Protect |
| Computer and Mathematical | 5,260,120 | 90.1 | 45.4 | 58.5 | Guard the pipeline |
| Architecture and Engineering | 2,602,660 | 64.8 | 54.6 | 63.6 | Strategic trap |
| Managerial | 1,596,600 | 66.5 | 53.4 | 60.8 | Strategic trap |
| Life, Physical, and Social Science | 1,291,170 | 63.4 | 53.7 | 73.4 | Strategic trap |
| Postsecondary Teaching | 642,420 | 79.8 | 44.0 | 68.3 | Guard the pipeline |
| Sales | 336,590 | 87.7 | 35.2 | 52.3 | Guard the pipeline |

Field is the leaf STEM discipline where an occupation has one, and the top-level
role type for the 31 — teaching, management, sales — that do not. An occupation
with several leaf memberships takes the lowest category id, so the assignment is
deterministic across runs.

**What this is not.** O\*NET carries no industry, clearance or criticality field,
so nothing here identifies an occupation as defence-relevant. The matrix scores
*properties* that make a handoff strategically dangerous, across the whole STEM
corpus. Which fields matter is the reader's overlay, not our measurement — and the
word "security" in the title describes the question being asked, not a
classification this dataset is able to make.

## 7. Validation

**Internal — 11 automated checks**, in `data/out/validation_report.json`. Errors
fail the run. They cover: every rostered occupation fetched; no swallowed
failures; parsed rows equal the page's own declared count; every task carries an
id; crosswalk ids resolve; activities present in the hierarchy; ratings inside
0–100; the wide table neither dropping nor duplicating a task; category
membership complete. Table grain is asserted at write time, so a fan-out in a
join raises instead of inflating counts.

**Convergent.** The index is computed only from the numeric dimensions, yet it
orders the model's *independently produced* categorical verdict monotonically:
`largely_automatable` 74.9 > `augmentable` 65.9 > `resistant` 43.6 >
`human_anchored` 40.3. Re-checked every run, with a warning if monotonicity breaks.

**External.** Benchmarked against Eloundou et al. (2023), whose `occ_level.csv` is
keyed to the same 8-digit O\*NET-SOC codes — **all 268 occupations matched, no
crosswalk needed**.

| Benchmark | Pearson | n |
| --- | --- | --- |
| **Human expert ratings, γ** | **0.850** | 268 |
| **Human expert ratings, β** | **0.834** | 268 |
| Human expert ratings, α (no tools) | 0.604 | 268 |
| GPT-4, β | 0.844 | 268 |
| Frey & Osborne (2017) | 0.027 | 150 |
| Felten, Raj & Seamans | −0.152 | 188 |
| Brynjolfsson/Mitchell/Rock SML | −0.103 | 188 |

Both halves matter. Agreement with human experts at r ≈ 0.85 is the evidence the
index measures what it claims. The weaker agreement with α is expected: α
excludes tools and this rubric explicitly asks what a model could do *given* them.

The pre-LLM measures disagree, and should. Frey & Osborne give mathematicians a
4.7% probability of computerisation; this index ranks them third of 268. Those
measures scored the routine/manual gradient; LLMs run the other way. **A strong
positive correlation there would have been the warning sign.** These three
expectations were written down as "moderate positive" before the test and the data
refuted them; the code records that.

### 7.1 Scoring reliability

Every number in this dataset came from a single scoring pass, so until now none
of it had a measure of its own stability. `reliability_report.json` closes that.

The catalogue of 963 subtasks has now been scored **three times** under the same
rubric (`2026-09-16.1`): twice by `claude-opus-5` and once by
`claude-sonnet-5`. Two passes of the same model isolate **sampling noise**; a
pass by a different model isolates how much of a score is the rubric rather than
the disposition of one model. Both are reported, because they license different
claims.

| | mean r | mean ICC | mean │d│ |
| --- | ---: | ---: | ---: |
| **Test-retest** — opus against opus | **0.972** | **0.985** | **3.8** |
| Cross-model — opus against sonnet | 0.934 | 0.948 | 7.9 |
| Cross-model — opus #2 against sonnet | 0.936 | 0.951 | 7.6 |

The two cross-model figures are independent measurements of the same quantity
and agree to within 0.002, which is some assurance that neither is an accident
of one run.

Cross-model, by dimension:

| Dimension | r | ICC | mean │d│ | within 10 | bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| Automation feasibility today | 0.908 | 0.952 | 6.6 | 83% | −0.2 |
| LLM exposure | 0.935 | 0.926 | 10.7 | 59% | −9.7 |
| Physical embodiment required | 0.963 | 0.966 | 8.4 | 76% | −7.6 |
| Interpersonal demand | 0.946 | 0.945 | 9.3 | 69% | −8.4 |
| Judgment under uncertainty | 0.920 | 0.948 | 6.2 | 87% | −3.7 |
| Accountability requirement | 0.932 | 0.951 | 6.9 | 82% | −4.9 |
| Error cost | 0.933 | 0.945 | 6.8 | 83% | −5.5 |

Both statistics are reported because Pearson's *r* is invariant to a constant
offset — two raters who differ by a flat twenty points correlate at 1.000 —
while ICC(2,1) (two-way random effects, absolute agreement) penalises exactly
that.

**The two measurements separate noise from calibration, which the cross-model
figure alone could not.** Two runs of the same model differ by 3.8 points on
average with a bias of at most 1.5 in any dimension; two different models differ
by 7.9 with a bias reaching 10.1 on `llm_exposure`. Decomposing the standard
deviations dimension by dimension:

| | mean sd, points |
| --- | ---: |
| Sampling noise — the same model against itself | 5.16 |
| Model difference — sonnet against the mean of both opus runs | 7.49 |

So roughly **41% of the disagreement between two different models is noise
either would show against itself**, and 59% is a real difference between them.
That is the number that was missing: the 9.7-point exposure offset in §6.6 is
now known to be mostly calibration rather than mostly chance.

**The rubric is doing the work.** Agreement at that level across two different
models is the strongest validity evidence in this document: the scores are a
property of the rubric and the activity text, not of one model's temperament.

**But the two models disagree about the level of the scale, not the ranking.**
Every dimension carries a negative bias — Sonnet reads lower — and on
`llm_exposure`, the dimension that drives the most, the gap is **9.7 points**.
The exposure distributions have almost the same shape (sd 23.9 against 23.4) and
a different centre (median 68 against 55). It is a calibration offset, not a
disagreement about which work is exposed.

**That offset does not move the indices, and does move the scenario counts.**

| Measure | Behaviour under the second pass |
| --- | --- |
| Susceptibility index | r = 0.951, mean shift −1.7 points |
| Automated subtasks, modest | **−51.4%** |
| Automated subtasks, substantial | **−34.9%** |
| Automated subtasks, extreme | **−25.9%** |

A continuous average absorbs a uniform shift; a hard threshold does not. The
scenario definitions in §6.6 cut on absolute exposure values (80, 70, 58), so a
ten-point shift in the level of the scale walks a large fraction of the
catalogue across a fixed line. **The scenario counts are therefore conditional
on the scoring model in a way the susceptibility index is not, and should be
read as such.**

**Both things were then fixed.** The scenario thresholds are now quantiles of
the corpus's own distribution rather than absolute points, exactly as §6.2 does
for the frontier. Measured across the two passes at task level, model
sensitivity falls from −50.5%, −36.6% and −28.0% to **+9.1%, +15.3% and +2.8%**;
what remains is genuine disagreement about ranking rather than about where the
scale sits. The quantiles are fixed constants inverted against the single-model
reference corpus, so on that corpus they reproduce the previous cuts to within
one task of 5,612.

And all three passes are **averaged into the canonical scores** rather than any
being discarded. `subtask_automation_scores.csv` carries `n_raters`, `raters`
and `score_disagreement` — the mean pairwise spread, and the project's only
per-subtask uncertainty measure. Across three raters it is **median 6.2 points,
p90 9.0, maximum 17.0**, tighter than the 7.6 / 11.9 / 22.9 of two raters
because the two same-model passes agree closely with each other.

Spearman–Brown puts the reliability of a k-rater mean at kr/(1+(k−1)r), which
at r = 0.934 gives 0.977 for three. **That figure is an upper bound rather than
a measurement**: two passes of the same model share whatever bias that model
has, so they are not three independent raters. The scores genuinely improve; the
number flatters itself.

What this moved, with the thresholds now adaptive:

| Figure | Single pass | Two-rater mean |
| --- | ---: | ---: |
| Susceptibility, occupation mean | 58.80 | 57.65 |
| Exposure, occupation mean | 63.82 | 58.47 |
| Automated tasks — modest | 1,193 | 1,215 |
| Automated tasks — substantial | 2,522 | 2,650 |
| Automated tasks — extreme | 3,852 | 3,864 |
| Handoff: human held / watch / crossing / handed off | 105 / 29 / 98 / 36 | 101 / 32 / 106 / 29 |

The exposure mean drops 5.4 points because it is now an average across two
differently-calibrated raters, and the scenario counts barely move *because* the
thresholds absorbed that recalibration. Under the old absolute cuts the same
change would have moved them by a quarter to a half. The frontier constants
adapted too — `frontier_k` from 53.0 to 51.1, the watch-point willingness gap
from 40 to 35.4 — which is why a few occupations changed handoff class.

### 7.2 Error bars

§7.1 measured how much a score moves between two runs of the same model. This
pushes that noise back through the derivation so the published figures carry
intervals and the published classifications carry stabilities.
`uncertainty_report.json` and `occupation_uncertainty.csv` hold the result.

**Method.** A parametric bootstrap: 400 resamples, seeded. Each resample jitters
every subtask score by its dimension's sampling sd, then re-runs the real
derivation — propagation to tasks and occupations, the susceptibility index, the
quadrant split, the handoff axes with their frontier recalibrated, and the
scenario cuts with theirs. An analytic interval is not available because the
quadrant and frontier thresholds are corpus medians and quantiles, which makes
every occupation's classification depend on every other occupation's score.

**The noise comes from the two same-model passes**, not from the three-rater
spread. Their disagreement is 59% calibration (§7.1), and calibration is not
something a re-run resamples. The difference of two independent draws has sd
√2·σ, so σ is sd(difference)/√2 — between 3.13 and 4.18 points depending on the
dimension, mean 3.65.

| Result | |
| --- | --- |
| Median 90% interval on occupation susceptibility | **1.8 points** (min 1.2, max 4.1) |
| Quadrant label holds in ≥90% of resamples | 235 of 268 |
| Quadrant label holds in ≥50% | 265 of 268 |
| Handoff class holds in ≥90% | 218 of 268 |
| Automated tasks, substantial | 2,585 · 90% interval 2,533–2,628 |

**The conclusions survive the noise.** A 1.8-point interval on a 0–100 index is
narrow enough that occupation rankings are not in question, and only three
occupations have a quadrant label that fails to survive half the resamples:
**Hydrologists** (35% — modal label *Insulated*, not the published
*Human-anchored*), **Zoologists and Wildlife Biologists** (40%) and **Medical
and Health Services Managers** (47% — modal *Displaceable*, not *Contested*).
Those three should not be cited at the quadrant level. Thirty-three more hold
between 50% and 90% of the time and deserve a hedge.

This corrects an estimate made before the bootstrap was run. A crude proxy —
jittering occupation-level values directly rather than resampling the subtask
scores and re-deriving — suggested 63 of 268 quadrant labels were coin-flips.
The real figure is three. The proxy was wrong in three ways at once: it used the
sd of the *difference* rather than σ, it applied that at occupation level where
the true noise has already averaged down through two aggregation steps, and it
used stale threshold constants. The lesson is the ordinary one — a shortcut
estimate of an uncertainty is itself uncertain, and in this case by a factor of
twenty.

**What this interval is not.** It is run-to-run variation in one model and
nothing else. It says nothing about whether the rubric asks the right questions,
whether the model reads the activity text correctly, or whether O\*NET's task
list is complete. Those are larger sources of error and none is quantified
anywhere in this document. Treat an interval here as a lower bound on how wrong
a number could be.

## 8. Limitations

1. **The scores are model judgment**, not survey data and not human expert
   judgment. They correlate with human ratings at r = 0.85, which is evidence of
   validity, not a substitute for it.
2. **No reliability estimate exists.** Nobody has re-run the scoring and measured
   agreement, so it is unknown whether the scores are a property of the work or of
   this model on this day. This is the largest open gap.
3. **Chunk-order effects are untested.** Activities were rated 12 per request; a
   score may depend partly on which others shared the request.
4. **The unit is the task, not the decision.** Watson's framework scores recurring
   decisions and is explicit that identifying which tasks are decisions is the
   novel work. That has not been done, so stage numbers are provisional.
5. **Activity assignment is analyst-coded**, not survey-measured. The network
   structure partly reflects O\*NET's vocabulary choices; the 27% of activities
   unique to one occupation may be analyst granularity as much as real
   specialisation.
6. **Exposure is not displacement.** Nothing here measures what employers will do,
   what regulation will permit, or how fast anything diffuses.
7. **O\*NET's own update cadence bounds the churn result** (§6.6).
8. **STEM only.** 287 occupations of O\*NET's ~1,000. Nothing here generalises to
   the rest of the labour market.
9. **Postsecondary teaching occupations are near-duplicates.** O\*NET gives all
   25-xxxx occupations one boilerplate profile; Economics and Political Science
   Teachers score cosine 1.00. They dominate any similarity ranking and are
   excluded from network defaults.
10. **Employment is a national cross-industry snapshot.** No geography, no
    industry detail, no projections.

## 9. Reproducibility

Every response is cached by URL digest with its fetch time and SHA-256, so a
rebuild is deterministic and needs no network:

```bash
python -m onet_scraper --offline --with-descriptors build
```

`data/out/manifest.json` records the tool version, source URLs, the O\*NET release,
per-file row counts and content hashes, run settings, and the validation summary.
The scroller's network layout is seeded, so the drawing is identical on every run.

A full rebuild from scratch is `python -m onet_scraper --refresh` followed by the
`score`, `report`, `employment`, `validate-external`, `pathways`, `churn`,
`figures`, `story`, `security` and `publish` stages. 250 tests run in CI on
Python 3.10 and 3.13, and a sixth job opens every published page in headless
Chrome and asserts it rendered.

**One caveat on the recorded stats.** `http_stats` in the manifest reads zero
because the final build was served entirely from cache; it counts the requests
that build made, not the requests that populated the cache.

Automation scores are tied to a rubric version (`2026-09-16.1`, recorded in
`manifest.json`). Scores produced under different rubric versions are not
comparable, so the version travels with the data.
