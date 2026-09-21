"""Tests for the network projection, index building and score propagation.

The scoring stage's API call cannot run offline, but everything around it can:
these exercise the maths that turns 963 rated subtasks into task- and
occupation-level measures.
"""

from __future__ import annotations

import statistics
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onet_scraper.descriptors import build_indices, build_related_occupations  # noqa: E402
from onet_scraper.network import (  # noqa: E402
    build_incidence,
    occupation_edges,
    occupation_nodes,
    subtask_network,
    validate_against_related,
    write_graphml,
)
from onet_scraper import security  # noqa: E402
from onet_scraper.score import estimate_cost, propagate, subtask_catalogue  # noqa: E402

TASKS = [
    {"onet_soc_code": "15-0001.00", "occupation_title": "A", "task_id": 1,
     "task": "t1", "task_category": "Core", "importance": 90.0},
    {"onet_soc_code": "15-0001.00", "occupation_title": "A", "task_id": 2,
     "task": "t2", "task_category": "Supplemental", "importance": 10.0},
    {"onet_soc_code": "29-0002.00", "occupation_title": "B", "task_id": 3,
     "task": "t3", "task_category": "Core", "importance": 50.0},
]
LINKS = [
    {"onet_soc_code": "15-0001.00", "task_id": 1, "dwa_id": "d1", "dwa_title": "D one"},
    {"onet_soc_code": "15-0001.00", "task_id": 1, "dwa_id": "d2", "dwa_title": "D two"},
    {"onet_soc_code": "15-0001.00", "task_id": 2, "dwa_id": "d3", "dwa_title": "D three"},
    {"onet_soc_code": "29-0002.00", "task_id": 3, "dwa_id": "d1", "dwa_title": "D one"},
    {"onet_soc_code": "29-0002.00", "task_id": 3, "dwa_id": "d2", "dwa_title": "D two"},
    {"onet_soc_code": "29-0002.00", "task_id": 3, "dwa_id": "d3", "dwa_title": "D three"},
]
OCCUPATIONS = [
    {"onet_soc_code": "15-0001.00", "title": "A", "stem_occupation_types": "x", "job_zone": 4},
    {"onet_soc_code": "29-0002.00", "title": "B", "stem_occupation_types": "y", "job_zone": 5},
]
HIERARCHY = [
    {"dwa_id": "d1", "dwa_title": "D one", "iwa_id": "i1", "iwa_title": "I one",
     "gwa_id": "g1", "gwa_title": "G one"},
]


class TestIncidence(unittest.TestCase):
    def test_importance_flows_to_subtasks(self):
        occ_dwa, weights, dwa_occ = build_incidence(LINKS, TASKS)
        self.assertEqual(occ_dwa["15-0001.00"], {"d1", "d2", "d3"})
        self.assertEqual(dwa_occ["d1"], {"15-0001.00", "29-0002.00"})
        # d1 comes from task 1 (importance 90), d3 from task 2 (importance 10)
        self.assertEqual(weights["15-0001.00"]["d1"], 90.0)
        self.assertEqual(weights["15-0001.00"]["d3"], 10.0)

    def test_exclude_soc_prefix(self):
        occ_dwa, _, _ = build_incidence(LINKS, TASKS, exclude_soc=("29",))
        self.assertEqual(set(occ_dwa), {"15-0001.00"})


class TestOccupationNetwork(unittest.TestCase):
    def setUp(self):
        self.occ_dwa, self.weights, self.dwa_occ = build_incidence(LINKS, TASKS)
        self.occs = {o["onet_soc_code"]: o for o in OCCUPATIONS}

    def test_identical_subtask_sets_give_cosine_one(self):
        edges = occupation_edges(self.occ_dwa, self.weights, self.occs, min_shared=3)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["shared_subtasks"], 3)
        self.assertAlmostEqual(edges[0]["cosine"], 1.0)
        self.assertAlmostEqual(edges[0]["jaccard"], 1.0)
        self.assertEqual(edges[0]["same_soc_major_group"], 0)

    def test_weighted_cosine_differs_from_unweighted(self):
        edges = occupation_edges(self.occ_dwa, self.weights, self.occs, min_shared=3)
        self.assertLess(edges[0]["weighted_cosine"], edges[0]["cosine"],
                        "importance weighting must distinguish these two occupations")

    def test_min_shared_filters(self):
        self.assertEqual(occupation_edges(self.occ_dwa, self.weights, self.occs,
                                          min_shared=4), [])

    def test_nodes_carry_degree(self):
        edges = occupation_edges(self.occ_dwa, self.weights, self.occs, min_shared=3)
        nodes = occupation_nodes(self.occ_dwa, self.weights, edges, self.occs)
        self.assertEqual({n["degree"] for n in nodes}, {1})
        self.assertEqual(nodes[0]["n_subtasks"], 3)

    def test_graphml_roundtrips_through_xml(self):
        import xml.etree.ElementTree as ET
        edges = occupation_edges(self.occ_dwa, self.weights, self.occs, min_shared=3)
        nodes = occupation_nodes(self.occ_dwa, self.weights, edges, self.occs)
        out = Path(__file__).parent / "_tmp.graphml"
        try:
            write_graphml(out, nodes, edges, "onet_soc_code", "cosine")
            root = ET.parse(out).getroot()
            ns = "{http://graphml.graphdrawing.org/xmlns}"
            self.assertEqual(len(root.findall(f"{ns}graph/{ns}node")), 2)
            self.assertEqual(len(root.findall(f"{ns}graph/{ns}edge")), 1)
        finally:
            out.unlink(missing_ok=True)


class TestSubtaskNetwork(unittest.TestCase):
    def test_co_occurrence(self):
        _, _, dwa_occ = build_incidence(LINKS, TASKS)
        edges, nodes = subtask_network(dwa_occ, HIERARCHY, LINKS, min_shared=2)
        self.assertEqual(len(edges), 3, "d1-d2, d1-d3, d2-d3 all co-occur in both jobs")
        self.assertEqual({n["dwa_id"] for n in nodes}, {"d1", "d2", "d3"})
        self.assertEqual(next(n for n in nodes if n["dwa_id"] == "d1")["gwa_title"], "G one")


class TestRelatednessValidation(unittest.TestCase):
    def test_recall_against_onet(self):
        occ_dwa, weights, _ = build_incidence(LINKS, TASKS)
        edges = occupation_edges(occ_dwa, weights, {}, min_shared=3)
        related = [{"onet_soc_code": "15-0001.00", "related_onet_soc_code": "29-0002.00",
                    "related_is_stem": 1}]
        result = validate_against_related(edges, related, top_k=5)
        self.assertTrue(result["available"])
        self.assertEqual(result["recall_at_k"], 1.0)

    def test_no_baseline_is_reported_not_crashed(self):
        self.assertEqual(validate_against_related([], []), {"available": False})


class TestScorePropagation(unittest.TestCase):
    def setUp(self):
        dims = {
            "automation_feasibility_today": 0, "llm_exposure": 0,
            "physical_embodiment_required": 0, "interpersonal_demand": 0,
            "judgment_under_uncertainty": 0, "accountability_requirement": 0,
            "error_cost": 0,
        }
        self.scores = [
            {**dims, "dwa_id": "d1", "llm_exposure": 100, "verdict": "largely_automatable"},
            {**dims, "dwa_id": "d2", "llm_exposure": 0, "verdict": "human_anchored"},
            {**dims, "dwa_id": "d3", "llm_exposure": 50, "verdict": "augmentable"},
        ]

    def test_task_score_is_mean_of_its_subtasks(self):
        task_rows, _ = propagate(self.scores, TASKS, LINKS, OCCUPATIONS)
        t1 = next(r for r in task_rows if r["task_id"] == 1)
        self.assertEqual(t1["n_subtasks_scored"], 2)
        self.assertEqual(t1["llm_exposure"], 50.0)  # mean of 100 and 0

    def test_occupation_score_is_importance_weighted(self):
        _, occ_rows = propagate(self.scores, TASKS, LINKS, OCCUPATIONS)
        a = next(r for r in occ_rows if r["onet_soc_code"] == "15-0001.00")
        # task1 (llm 50, importance 90) and task2 (llm 50, importance 10) -> 50
        self.assertEqual(a["llm_exposure"], 50.0)
        b = next(r for r in occ_rows if r["onet_soc_code"] == "29-0002.00")
        self.assertEqual(b["llm_exposure"], 50.0)  # mean of 100, 0, 50
        self.assertEqual(b["task_coverage"], 1.0)

    def test_weighting_actually_bites(self):
        """A high-importance automatable task should dominate a trivial one."""
        tasks = [
            {"onet_soc_code": "15-0001.00", "occupation_title": "A", "task_id": 1,
             "task": "t1", "task_category": "Core", "importance": 99.0},
            {"onet_soc_code": "15-0001.00", "occupation_title": "A", "task_id": 2,
             "task": "t2", "task_category": "Core", "importance": 1.0},
        ]
        links = [
            {"onet_soc_code": "15-0001.00", "task_id": 1, "dwa_id": "d1", "dwa_title": ""},
            {"onet_soc_code": "15-0001.00", "task_id": 2, "dwa_id": "d2", "dwa_title": ""},
        ]
        _, occ = propagate(self.scores, tasks, links, OCCUPATIONS[:1])
        self.assertGreater(occ[0]["llm_exposure"], 95.0,
                           "the importance-99 task must dominate the importance-1 one")

    def test_unscored_subtasks_are_skipped_not_zeroed(self):
        task_rows, _ = propagate([self.scores[0]], TASKS, LINKS, OCCUPATIONS)
        t2 = [r for r in task_rows if r["task_id"] == 2]
        self.assertEqual(t2, [], "a task with no scored subtask must be absent, not 0")


class TestCatalogueAndCost(unittest.TestCase):
    def test_catalogue_is_deduplicated(self):
        catalogue = subtask_catalogue(LINKS, HIERARCHY)
        self.assertEqual(len(catalogue), 3)
        d1 = next(c for c in catalogue if c["dwa_id"] == "d1")
        self.assertEqual(d1["n_occupations"], 2)
        self.assertEqual(d1["gwa_title"], "G one")

    def test_cost_scales_with_chunk_size(self):
        small = estimate_cost([{"dwa_id": str(i)} for i in range(100)], 5, "claude-opus-5")
        large = estimate_cost([{"dwa_id": str(i)} for i in range(100)], 25, "claude-opus-5")
        self.assertEqual(small["requests"], 20)
        self.assertEqual(large["requests"], 4)
        self.assertLess(large["est_cost_usd"], small["est_cost_usd"])


class TestIndices(unittest.TestCase):
    def test_missing_element_is_reported(self):
        tables = {"work_context.csv": [
            {"onet_soc_code": "15-0001.00", "element_name": "Contact With Others",
             "scale_id": "CX", "data_value": 5.0},
            {"onet_soc_code": "29-0002.00", "element_name": "Contact With Others",
             "scale_id": "CX", "data_value": 1.0},
        ]}
        rows, missing = build_indices(tables, OCCUPATIONS)
        self.assertTrue(any("Work With or Contribute" in m for m in missing),
                        "an element absent from the data must be reported, not silently zero")
        a = next(r for r in rows if r["onet_soc_code"] == "15-0001.00")
        self.assertEqual(a["collaboration"], 100.0)  # rescaled max of the observed range
        self.assertIsNone(a["bottleneck_creative_intelligence"])

    def test_related_occupations_flags_stem(self):
        rows = build_related_occupations(
            [{"O*NET-SOC Code": "15-0001.00", "Related O*NET-SOC Code": "29-0002.00",
              "Related Title": "B", "Relatedness Tier": "Primary-Short", "Index": "1"}],
            {"15-0001.00", "29-0002.00"})
        self.assertEqual(rows[0]["related_is_stem"], 1)


class TestSecurityMatrix(unittest.TestCase):
    """The efficiency / removal-risk / reconstitution matrix."""

    def _task(self, **kw):
        base = {"onet_soc_code": "15-0001.00", "task_id": 1, "importance": 50.0,
                "modest": "unchanged", "substantial": "unchanged",
                "extreme": "unchanged", "error_cost": 50.0,
                "accountability_requirement": 50.0,
                "judgment_under_uncertainty": 50.0,
                "interpersonal_demand": 50.0,
                "physical_embodiment_required": 50.0}
        base.update(kw)
        return base

    def test_efficiency_credits_automated_fully_and_augmented_partly(self):
        auto = [self._task(substantial="automated")]
        aug = [self._task(substantial="augmented")]
        none = [self._task(substantial="unchanged")]
        self.assertEqual(security.efficiency(auto, "substantial"), 100.0)
        self.assertEqual(security.efficiency(none, "substantial"), 0.0)
        self.assertEqual(security.efficiency(aug, "substantial"),
                         100.0 * security.AUGMENTED_CREDIT)

    def test_efficiency_is_importance_weighted_not_a_task_count(self):
        # One trivial automatable task plus one dominant task that stays human
        # is not an efficiency opportunity, even though it is 50% of the tasks.
        tasks = [self._task(task_id=1, importance=5.0, substantial="automated"),
                 self._task(task_id=2, importance=95.0, substantial="unchanged")]
        self.assertEqual(security.efficiency(tasks, "substantial"), 5.0)

    def test_efficiency_rises_with_scenario_severity(self):
        tasks = [self._task(task_id=1, modest="unchanged",
                            substantial="augmented", extreme="automated")]
        modest = security.efficiency(tasks, "modest")
        substantial = security.efficiency(tasks, "substantial")
        extreme = security.efficiency(tasks, "extreme")
        self.assertLess(modest, substantial)
        self.assertLess(substantial, extreme)

    def test_removal_risk_ignores_interpersonal_and_physical(self):
        """The central design claim: this is a risk index, not an
        anchoring index. Two tasks differing only in interpersonal demand and
        physical embodiment must score identically."""
        low = [self._task(interpersonal_demand=0.0, physical_embodiment_required=0.0)]
        high = [self._task(interpersonal_demand=100.0,
                           physical_embodiment_required=100.0)]
        self.assertEqual(security.removal_risk(low), security.removal_risk(high))

    def test_removal_risk_weights_error_cost_highest(self):
        base = self._task(error_cost=0.0, accountability_requirement=0.0,
                          judgment_under_uncertainty=0.0)
        bump_error = security.removal_risk([dict(base, error_cost=100.0)])
        bump_acct = security.removal_risk(
            [dict(base, accountability_requirement=100.0)])
        bump_judge = security.removal_risk(
            [dict(base, judgment_under_uncertainty=100.0)])
        self.assertGreater(bump_error, bump_acct)
        self.assertGreater(bump_acct, bump_judge)
        self.assertAlmostEqual(
            sum(security.RISK_WEIGHTS.values()), 1.0, places=6)

    def test_risk_at_stake_only_counts_handed_over_work(self):
        # The risky task stays human; the automated task is harmless. Overall
        # risk is high, but the risk actually being handed over is not.
        tasks = [self._task(task_id=1, importance=50.0, error_cost=100.0,
                            accountability_requirement=100.0,
                            judgment_under_uncertainty=100.0,
                            substantial="unchanged"),
                 self._task(task_id=2, importance=50.0, error_cost=0.0,
                            accountability_requirement=0.0,
                            judgment_under_uncertainty=0.0,
                            substantial="automated")]
        self.assertAlmostEqual(security.removal_risk(tasks), 50.0, places=1)
        self.assertEqual(security.risk_at_stake(tasks, "substantial"), 0.0)

    def test_risk_at_stake_is_zero_when_nothing_is_handed_over(self):
        tasks = [self._task(substantial="unchanged")]
        self.assertEqual(security.risk_at_stake(tasks, "substantial"), 0.0)

    def test_reconstitution_rises_with_training_depth(self):
        shallow = security.reconstitution(3, 10000, 0.3, 3.0, 6.0)
        deep = security.reconstitution(5, 10000, 0.3, 3.0, 6.0)
        self.assertLess(shallow["reconstitution"], deep["reconstitution"])
        self.assertEqual(deep["training_depth"], 100.0)

    def test_reconstitution_treats_small_cohorts_as_harder(self):
        tiny = security.reconstitution(4, 1_000, 0.3, 3.0, 6.0)
        huge = security.reconstitution(4, 1_000_000, 0.3, 3.0, 6.0)
        self.assertGreater(tiny["scarcity"], huge["scarcity"])
        self.assertGreater(tiny["reconstitution"], huge["reconstitution"])

    def test_scarcity_is_log_scaled(self):
        """A linear scale would call everything except the largest occupation
        scarce, because the corpus spans four orders of magnitude."""
        mid = security._scarcity(10**4.5, 3.0, 6.0)
        self.assertAlmostEqual(mid, 50.0, places=1)

    def test_isolation_is_the_inverse_of_shared_activity(self):
        shared = security.reconstitution(4, 10000, 0.9, 3.0, 6.0)
        isolated = security.reconstitution(4, 10000, 0.1, 3.0, 6.0)
        self.assertLess(shared["isolation"], isolated["isolation"])
        self.assertLess(shared["reconstitution"], isolated["reconstitution"])

    def test_missing_inputs_fall_back_to_the_midpoint(self):
        parts = security.reconstitution(None, None, None, 3.0, 6.0)
        self.assertEqual(parts["scarcity"], security.MIDPOINT)
        self.assertEqual(parts["isolation"], security.MIDPOINT)
        self.assertEqual(parts["reconstitution"], security.MIDPOINT)

    def test_all_eight_octants_are_reachable_and_named(self):
        seen = set()
        for eff in (10.0, 90.0):
            for risk in (10.0, 90.0):
                for recon in (10.0, 90.0):
                    seen.add(security.octant(eff, risk, recon)[0])
        self.assertEqual(len(seen), 8)
        self.assertEqual(len(security.OCTANTS), 8)
        self.assertEqual(len({n for n, _ in security.OCTANTS.values()}), 8)

    def test_octant_boundary_is_inclusive_at_the_midpoint(self):
        self.assertEqual(security.octant(50.0, 50.0, 50.0)[0], "Strategic trap")
        self.assertEqual(security.octant(49.9, 49.9, 49.9)[0], "Low stakes")

    def test_trap_score_is_a_product_so_one_low_axis_demotes(self):
        """A sum would rank overwhelming-efficiency-no-risk above
        dangerous-on-all-three, which inverts what a planner needs to read."""
        lopsided = security.trap_score(100.0, 5.0, 5.0)
        balanced = security.trap_score(60.0, 60.0, 60.0)
        self.assertGreater(balanced, lopsided)

    def test_field_map_prefers_a_leaf_discipline_over_a_role_type(self):
        cats = [{"stem_category_id": "1", "stem_category_name": "Role", "is_leaf": "0"},
                {"stem_category_id": "1-2", "stem_category_name": "Discipline",
                 "is_leaf": "1"}]
        members = [{"onet_soc_code": "X", "stem_category_id": "1"},
                   {"onet_soc_code": "X", "stem_category_id": "1-2"}]
        self.assertEqual(security.field_map(members, cats)["X"], "Discipline")
        # reversed input order must not change the answer
        self.assertEqual(
            security.field_map(list(reversed(members)), cats)["X"], "Discipline")

    def test_field_map_falls_back_to_role_type_when_no_discipline(self):
        cats = [{"stem_category_id": "4", "stem_category_name": "Managerial",
                 "is_leaf": "0"}]
        members = [{"onet_soc_code": "X", "stem_category_id": "4"}]
        self.assertEqual(security.field_map(members, cats)["X"], "Managerial")

    def test_field_map_is_deterministic_with_two_disciplines(self):
        cats = [{"stem_category_id": "1-2", "stem_category_name": "Comp",
                 "is_leaf": "1"},
                {"stem_category_id": "1-4", "stem_category_name": "Science",
                 "is_leaf": "1"}]
        members = [{"onet_soc_code": "X", "stem_category_id": "1-4"},
                   {"onet_soc_code": "X", "stem_category_id": "1-2"}]
        self.assertEqual(security.field_map(members, cats)["X"], "Comp")

    # -- the double-count regression -------------------------------------
    def _corpus(self):
        fates = [
            {"onet_soc_code": "29-1141.00", "task_id": 1, "importance": 90.0,
             "modest": "unchanged", "substantial": "automated",
             "extreme": "automated"},
            {"onet_soc_code": "29-1141.01", "task_id": 2, "importance": 90.0,
             "modest": "unchanged", "substantial": "automated",
             "extreme": "automated"},
        ]
        scores = [
            {"task_id": 1, "error_cost": 90.0, "accountability_requirement": 90.0,
             "judgment_under_uncertainty": 90.0},
            {"task_id": 2, "error_cost": 90.0, "accountability_requirement": 90.0,
             "judgment_under_uncertainty": 90.0},
        ]
        occs = [
            {"onet_soc_code": "29-1141.00", "title": "Nurses", "job_zone": 4,
             "stem_occupation_types": "Healthcare"},
            {"onet_soc_code": "29-1141.01", "title": "Acute Nurses", "job_zone": 4,
             "stem_occupation_types": "Healthcare"},
        ]
        # Both O*NET codes roll up to one SOC and each carries its full figure.
        emp = {"29-1141.00": 3_000_000.0, "29-1141.01": 3_000_000.0}
        soc_of = {"29-1141.00": "29-1141", "29-1141.01": "29-1141"}
        fields = {"29-1141.00": "Healthcare", "29-1141.01": "Healthcare"}
        return fates, scores, occs, emp, soc_of, fields

    def test_field_employment_collapses_to_soc_before_summing(self):
        """Both O*NET codes roll up to one SOC and each carries its full figure.
        Knowing the SOC map at build time is what prevents the double count -
        without it every occupation looks like its own SOC."""
        fates, scores, occs, emp, soc_of, fields = self._corpus()
        correct = security.by_field(
            security.build(fates, scores, occs, emp, fields=fields, soc_of=soc_of))
        naive = security.by_field(
            security.build(fates, scores, occs, emp, fields=fields))
        one = [r for r in correct if r["scenario"] == "substantial"][0]
        two = [r for r in naive if r["scenario"] == "substantial"][0]
        self.assertEqual(one["total_employment"], 3_000_000.0)
        self.assertEqual(two["total_employment"], 6_000_000.0)
        self.assertNotEqual(one["total_employment"], two["total_employment"])

    def test_summary_employment_collapses_to_soc(self):
        fates, scores, occs, emp, soc_of, fields = self._corpus()
        rows = security.build(fates, scores, occs, emp, fields=fields, soc_of=soc_of)
        report = security.summarise(rows, soc_of)
        self.assertEqual(
            report["scenarios"]["substantial"]["total_employment"], 3_000_000.0)
        shares = [c["share_employment"]
                  for c in report["scenarios"]["substantial"]["by_octant"].values()]
        self.assertAlmostEqual(sum(shares), 1.0, places=3)

    def test_build_emits_one_row_per_occupation_per_scenario(self):
        fates, scores, occs, emp, soc_of, fields = self._corpus()
        rows = security.build(fates, scores, occs, emp, fields=fields)
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["scenario"] for r in rows},
                         {"modest", "substantial", "extreme"})
        self.assertEqual({r["field"] for r in rows}, {"Healthcare"})

    def test_build_skips_occupations_with_no_scored_tasks(self):
        fates, scores, occs, emp, soc_of, fields = self._corpus()
        occs.append({"onet_soc_code": "99-9999.00", "title": "Ghost",
                     "job_zone": 4, "stem_occupation_types": "Healthcare"})
        rows = security.build(fates, scores, occs, emp, fields=fields)
        self.assertNotIn("99-9999.00", {r["onet_soc_code"] for r in rows})

    def test_ranking_is_insensitive_to_the_augmented_credit(self):
        """AUGMENTED_CREDIT is a round number standing in for 'some'. The
        ordering it produces must not depend on the exact value."""
        tasks_a = [self._task(task_id=1, importance=80.0, substantial="augmented")]
        tasks_b = [self._task(task_id=2, importance=80.0, substantial="automated")]
        original = security.AUGMENTED_CREDIT
        try:
            for credit in (0.35, 0.5, 0.65):
                security.AUGMENTED_CREDIT = credit
                self.assertLess(security.efficiency(tasks_a, "substantial"),
                                security.efficiency(tasks_b, "substantial"))
        finally:
            security.AUGMENTED_CREDIT = original


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestSusceptibilityIndex(unittest.TestCase):
    """The index must not double-count the two near-collinear dimension pairs."""

    def _row(self, **kw):
        base = {d: 0 for d in (
            "automation_feasibility_today", "llm_exposure", "physical_embodiment_required",
            "interpersonal_demand", "judgment_under_uncertainty",
            "accountability_requirement", "error_cost")}
        base.update(kw)
        return base

    def test_stakes_pair_collapses_to_one_factor(self):
        from onet_scraper.susceptibility import add_axes
        # accountability and error_cost correlate at r=0.90; together they must carry
        # the weight of one anchoring component, not two.
        both = add_axes(self._row(accountability_requirement=100, error_cost=100))
        one = add_axes(self._row(interpersonal_demand=100))
        self.assertEqual(both["anchoring"], one["anchoring"],
                         "the stakes pair must weigh the same as any single component")

    def test_physical_not_counted_twice(self):
        from onet_scraper.susceptibility import add_axes
        # physical_embodiment is llm_exposure's mirror (r=-0.89); it belongs to
        # anchoring only, and must not also be subtracted from exposure.
        r = add_axes(self._row(llm_exposure=80, physical_embodiment_required=100))
        self.assertEqual(r["exposure"], 80.0)

    def test_susceptibility_direction_and_clamp(self):
        from onet_scraper.susceptibility import add_axes
        hi = add_axes(self._row(llm_exposure=100))
        lo = add_axes(self._row(llm_exposure=0, interpersonal_demand=100,
                                accountability_requirement=100, error_cost=100,
                                judgment_under_uncertainty=100,
                                physical_embodiment_required=100))
        self.assertGreater(hi["susceptibility"], lo["susceptibility"])
        for r in (hi, lo):
            self.assertGreaterEqual(r["susceptibility"], 0)
            self.assertLessEqual(r["susceptibility"], 100)

    def test_deployment_gap(self):
        from onet_scraper.susceptibility import add_axes
        r = add_axes(self._row(llm_exposure=90, automation_feasibility_today=20))
        self.assertEqual(r["deployment_gap"], 70.0)

    def test_quadrant_naming(self):
        from onet_scraper.susceptibility import quadrant
        self.assertEqual(quadrant(80, 20, 50, 50), "Displaceable")
        self.assertEqual(quadrant(80, 80, 50, 50), "Contested")
        self.assertEqual(quadrant(20, 80, 50, 50), "Human-anchored")
        self.assertEqual(quadrant(20, 20, 50, 50), "Insulated")

    def test_convergent_validity_flags_non_monotonic(self):
        from onet_scraper.susceptibility import convergent_validity
        bad = [{"verdict": "largely_automatable", "susceptibility": 10.0},
               {"verdict": "human_anchored", "susceptibility": 90.0}]
        self.assertFalse(convergent_validity(bad)["monotonic"])
        good = [{"verdict": "largely_automatable", "susceptibility": 90.0},
                {"verdict": "human_anchored", "susceptibility": 10.0}]
        self.assertTrue(convergent_validity(good)["monotonic"])


class TestEmploymentJoin(unittest.TestCase):
    """The SOC join is where a naive implementation silently multi-counts."""

    OEWS = {
        "29-1141": {"soc_code": "29-1141", "soc_title": "Registered Nurses",
                    "total_employment": 3_000_000.0, "annual_mean_wage": 90_000.0,
                    "annual_median_wage": 86_000.0},
        "29-2010": {"soc_code": "29-2010", "soc_title": "Clinical Lab Techs",
                    "total_employment": 300_000.0, "annual_mean_wage": 60_000.0,
                    "annual_median_wage": 58_000.0},
        "15-2051": {"soc_code": "15-2051", "soc_title": "Data Scientists",
                    "total_employment": 200_000.0, "annual_mean_wage": 120_000.0,
                    "annual_median_wage": 110_000.0},
    }

    def _occ(self, code, susc, quad="Displaceable"):
        return {"onet_soc_code": code, "title": code, "susceptibility": susc,
                "exposure": susc, "anchoring": 100 - susc, "deployment_gap": 10,
                "quadrant": quad, "stem_occupation_types": "x"}

    def test_five_onet_nurses_count_three_million_workers_once(self):
        from onet_scraper.employment import build_soc_table
        occ = [self._occ(f"29-1141.0{i}", 50, "Human-anchored") for i in range(5)]
        rows, diag = build_soc_table(occ, self.OEWS)
        self.assertEqual(len(rows), 1, "five O*NET nurse occupations = one SOC row")
        self.assertEqual(rows[0]["total_employment"], 3_000_000.0)
        self.assertEqual(rows[0]["n_onet_occupations"], 5)
        self.assertEqual(diag["employment_covered"], 3_000_000.0,
                         "employment must be counted once, not five times")

    def test_broad_code_fallback_does_not_double_count(self):
        from onet_scraper.employment import build_soc_table
        # 29-2011 and 29-2012 are both published by BLS as 29-2010.
        occ = [self._occ("29-2011.00", 60), self._occ("29-2012.00", 40)]
        rows, diag = build_soc_table(occ, self.OEWS)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["soc_code"], "29-2010")
        self.assertEqual(diag["employment_covered"], 300_000.0)
        self.assertEqual(rows[0]["susceptibility"], 50.0, "mean of 60 and 40")

    def test_within_soc_disagreement_is_reported(self):
        from onet_scraper.employment import build_soc_table
        rows, _ = build_soc_table(
            [self._occ("29-1141.01", 20), self._occ("29-1141.02", 80)], self.OEWS)
        self.assertGreater(rows[0]["susceptibility_sd_within_soc"], 40,
                           "a hidden split inside one SOC must be visible")

    def test_unmatched_codes_are_reported_not_dropped_silently(self):
        from onet_scraper.employment import build_soc_table
        rows, diag = build_soc_table([self._occ("99-9999.00", 50)], self.OEWS)
        self.assertEqual(rows, [])
        self.assertEqual(diag["unmatched_onet_codes"], ["99-9999.00"])

    def test_resolve_soc(self):
        from onet_scraper.employment import resolve_soc
        avail = set(self.OEWS)
        self.assertEqual(resolve_soc("15-2051.00", avail), "15-2051")
        self.assertEqual(resolve_soc("29-2011.00", avail), "29-2010")
        self.assertIsNone(resolve_soc("99-9999.00", avail))

    def test_bls_sentinels_are_not_parsed_as_numbers(self):
        from onet_scraper.employment import _num
        for marker in ("*", "**", "#", "~", ""):
            self.assertIsNone(_num(marker), f"{marker!r} is a suppression flag, not a value")
        self.assertEqual(_num("1,234"), 1234.0)
        self.assertEqual(_num(56.7), 56.7)

    def test_headline_weighting(self):
        from onet_scraper.employment import build_soc_table, headline_stats
        occ = [self._occ("29-1141.00", 20, "Human-anchored"),
               self._occ("15-2051.00", 90, "Displaceable")]
        rows, _ = build_soc_table(occ, self.OEWS)
        stats = headline_stats(rows)
        # 3.0M at 20 and 0.2M at 90 -> weighted mean must sit near 20, not 55.
        self.assertLess(stats["employment_weighted_susceptibility"], 30)
        self.assertEqual(stats["unweighted_susceptibility"], 55.0)
        self.assertLess(stats["weighting_shifts_result_by"], -20)
        self.assertEqual(stats["by_quadrant"]["Human-anchored"]["employment"], 3_000_000.0)


class TestExternalValidation(unittest.TestCase):
    def test_pearson_and_spearman(self):
        from onet_scraper.external import pearson, spearman
        a = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(pearson(a, [2, 4, 6, 8, 10]), 1.0)
        self.assertAlmostEqual(pearson(a, [10, 8, 6, 4, 2]), -1.0)
        # Spearman is rank-based, so a monotone non-linear map is still 1.0
        self.assertAlmostEqual(spearman(a, [1, 4, 9, 16, 25]), 1.0)
        self.assertLess(pearson(a, [1, 4, 9, 16, 25]), 1.0)

    def test_spearman_handles_ties(self):
        from onet_scraper.external import spearman
        self.assertAlmostEqual(spearman([1, 1, 2, 2], [1, 1, 2, 2]), 1.0)

    def test_join_keeps_soc_level_measures(self):
        from onet_scraper.external import join
        ours = [{"onet_soc_code": "15-2021.00", "title": "Mathematicians",
                 "susceptibility": 77.4, "exposure": 80.0, "anchoring": 25.0}]
        bench = {"occ": {"15-2021.00": {"human_rating_beta": "0.8",
                                        "dv_rating_beta": "0.75"}},
                 "auto": {"15-2021": {"freyOsborne": "0.047", "mSML": "3.1",
                                      "felten_raj_seamans": "4.2"}}}
        rows, cov = join(ours, bench)
        self.assertEqual(cov["matched"], 1)
        self.assertEqual(rows[0]["human_beta"], 0.8)
        self.assertEqual(rows[0]["frey_osborne"], 0.047,
                         "SOC-level measures must join on the 6-digit code")

    def test_unmatched_reported(self):
        from onet_scraper.external import join
        rows, cov = join([{"onet_soc_code": "99-9999.00", "susceptibility": 50}],
                         {"occ": {}, "auto": {}})
        self.assertEqual(rows, [])
        self.assertEqual(cov["unmatched"], ["99-9999.00"])

    def test_correlate_skips_thin_overlap(self):
        from onet_scraper.external import correlate
        rows = [{"our_susceptibility": i, "human_beta": i} for i in range(5)]
        result = {c["measure"]: c for c in correlate(rows)}
        self.assertIsNone(result["human_beta"]["pearson"],
                          "fewer than 10 pairs must not produce a correlation")


class TestHandoffFramework(unittest.TestCase):
    """Watson's two axes, six stages and four categories."""

    def _row(self, **kw):
        base = {d: 0 for d in (
            "automation_feasibility_today", "llm_exposure", "physical_embodiment_required",
            "interpersonal_demand", "judgment_under_uncertainty",
            "accountability_requirement", "error_cost")}
        base.update(kw)
        base.setdefault("onet_soc_code", "15-0001.00")
        return base

    def test_axes_separate_can_from_permitted(self):
        from onet_scraper.handoff import axes
        # Pure capability, nothing in the way.
        t, r = axes(self._row(llm_exposure=100))
        self.assertGreater(t, 90)
        self.assertEqual(r, 0.0)
        # Pure resistance: accountability and stakes, no capability.
        t, r = axes(self._row(accountability_requirement=100, error_cost=100,
                              interpersonal_demand=100))
        self.assertEqual(r, 100.0)

    def test_stage_is_the_lower_of_the_two_ceilings(self):
        from onet_scraper.handoff import stage
        # Full capability but total resistance -> nothing is permitted.
        self.assertEqual(stage(100, 100), 0)
        # Full capability, no resistance -> AI led, unreviewed.
        self.assertEqual(stage(100, 0), 5)
        # Capability is the binding constraint here, not permission.
        self.assertEqual(stage(20, 0), 1)

    def test_low_tractability_is_never_handed_off(self):
        """The bug the frontier curve alone produces: a constant-product curve
        puts 'low on both axes' on the same side as 'high T, low R'."""
        from onet_scraper.handoff import classify
        self.assertEqual(classify(33, 45, 10), "Human held")
        self.assertEqual(classify(49, 20, 10), "Human held")

    def test_handed_off_and_human_held_separate(self):
        from onet_scraper.handoff import classify
        self.assertEqual(classify(78, 30, 5), "Handed off")
        self.assertEqual(classify(55, 80, 5), "Human held")

    def test_watch_point_needs_all_three_conditions(self):
        from onet_scraper.handoff import classify
        self.assertEqual(classify(64, 70, 50), "Watch point")
        # capability present and resistance high, but no willingness gap
        self.assertNotEqual(classify(64, 70, 5), "Watch point")
        # wide gap but the work is not tractable
        self.assertEqual(classify(40, 70, 50), "Human held")

    def test_frontier_is_monotone_decreasing(self):
        from onet_scraper.handoff import frontier_resistance
        values = [frontier_resistance(t) for t in (40, 50, 60, 70, 80)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_build_reports_pending_crossings_and_weighty_ones(self):
        from onet_scraper.handoff import WEIGHTY_CROSSINGS, build
        rows = build([self._row(llm_exposure=80, automation_feasibility_today=20,
                                accountability_requirement=30, error_cost=30)])
        row = rows[0]
        self.assertGreater(row["stage_reachable"], row["stage_now"])
        self.assertEqual(row["pending_crossings"],
                         row["stage_reachable"] - row["stage_now"])
        self.assertEqual(row["willingness_gap"], 60.0)
        if row["stage_now"] in WEIGHTY_CROSSINGS:
            self.assertTrue(row["weighty_crossing"])

    def test_no_pending_crossing_means_no_weighty_label(self):
        from onet_scraper.handoff import build
        rows = build([self._row(llm_exposure=50, automation_feasibility_today=50)])
        self.assertEqual(rows[0]["pending_crossings"], 0)
        self.assertEqual(rows[0]["weighty_crossing"], "")

    def test_summarise_counts_employment_once(self):
        from onet_scraper.handoff import build, summarise
        rows = build(
            [self._row(onet_soc_code="29-1141.01", llm_exposure=70,
                       accountability_requirement=70, error_cost=70),
             self._row(onet_soc_code="29-1141.02", llm_exposure=70,
                       accountability_requirement=70, error_cost=70)],
            employment={"29-1141.01": 100.0, "29-1141.02": 100.0})
        s = summarise(rows)
        self.assertEqual(s["occupations"], 2)
        self.assertIn("by_classification", s)


class TestWageProtection(unittest.TestCase):
    def _soc(self, code, emp, wage, expo, anch):
        return {"soc_code": code, "soc_title": code, "total_employment": emp,
                "annual_mean_wage": wage, "exposure": expo, "anchoring": anch,
                "susceptibility": 50 + (expo - anch) / 2, "quadrant": ""}

    def test_returns_the_same_shape_when_empty(self):
        """The success and failure paths must agree, or a caller's availability
        check silently skips the whole analysis."""
        from onet_scraper.wages import build
        empty, full = build([]), build([self._soc("a", 100, 50000, 60, 40)])
        self.assertEqual(set(empty), set(full))
        self.assertFalse(empty["available"])
        self.assertTrue(full["available"])

    def test_deciles_hold_workers_constant_not_occupations(self):
        """One occupation larger than a decile must be split across buckets,
        not dropped whole into one - otherwise the buckets are not deciles."""
        from onet_scraper.wages import build
        rows = [self._soc("big", 9000, 40000, 50, 50),
                self._soc("small", 1000, 200000, 90, 10)]
        d = build(rows)["deciles"]
        self.assertEqual(len(d), 10)
        sizes = [x["workers"] for x in d]
        self.assertLess(max(sizes) - min(sizes), 2, "every decile holds 1,000 workers")
        # the cheap occupation fills the first nine buckets
        self.assertEqual(sum(1 for x in d if x["mean_wage"] == 40000), 9)

    def test_protection_classes(self):
        from onet_scraper.wages import classify_protection
        self.assertEqual(classify_protection(80, 80, 60, 50),
                         "Accountability - AI could, a human must answer")
        self.assertEqual(classify_protection(80, 20, 60, 50),
                         "Unprotected - exposed and lightly anchored")
        self.assertEqual(classify_protection(20, 80, 60, 50),
                         "Capability - AI cannot do much of it")


class TestTransitions(unittest.TestCase):
    def _setup(self):
        occ = [{"onet_soc_code": "A", "title": "A", "susceptibility": 80},
               {"onet_soc_code": "B", "title": "B", "susceptibility": 40},
               {"onet_soc_code": "C", "title": "C", "susceptibility": 78}]
        edges = [{"source": "A", "target": "B", "cosine": 0.5},
                 {"source": "A", "target": "C", "cosine": 0.9}]
        occ_dwa = {"A": {"d1", "d2"}, "B": {"d1", "d3"}, "C": {"d1", "d2"}}
        dwa_susc = {"d1": 90.0, "d2": 85.0, "d3": 20.0}
        return edges, occ, occ_dwa, dwa_susc, {}

    def test_similar_but_equally_exposed_is_not_a_destination(self):
        from onet_scraper.transitions import build
        moves, stranded, s = build(*self._setup())
        a = next(m for m in moves if m["onet_soc_code"] == "A")
        # C is the closest neighbour but offers no relief; B must win
        self.assertEqual(a["destination_code"], "B")

    def test_stranded_when_every_neighbour_is_exposed(self):
        from onet_scraper.transitions import build
        edges, occ, occ_dwa, dwa_susc, emp = self._setup()
        edges = [{"source": "A", "target": "C", "cosine": 0.9}]
        moves, stranded, s = build(edges, occ, occ_dwa, dwa_susc, emp)
        self.assertIn("A", [r["onet_soc_code"] for r in stranded])
        self.assertEqual(s["stranded_share"], round(len(stranded)/3, 3))

    def test_sweep_reports_threshold_dependence(self):
        from onet_scraper.transitions import sweep
        out = sweep(*self._setup())
        self.assertEqual(len(out), 9)
        self.assertTrue(all("stranded" in r for r in out))


class TestChurn(unittest.TestCase):
    def test_picks_the_real_task_file_not_the_green_subset(self):
        """A suffix match also catches 'Green Task Statements.txt', which is a
        small subset that parses cleanly and silently replaces the real file."""
        import io, zipfile
        from unittest.mock import Mock
        from onet_scraper.churn import fetch_release
        buf = io.BytesIO()
        hdr = "O*NET-SOC Code\tTitle\tTask ID\tTask\n"
        with zipfile.ZipFile(buf, "w") as z:
            # green file first, exactly as the real archives order them
            z.writestr("db/Green Task Statements.txt", hdr + "11-0000.00\tG\t1\tgreen task\n")
            z.writestr("db/Task Statements.txt",
                       hdr + "11-0000.00\tR\t2\treal task\n15-0000.00\tR\t3\tanother\n")
        client = Mock()
        client.get.return_value = Mock(content=buf.getvalue())
        out = fetch_release(client, "24_0")
        self.assertEqual(len(out), 2, "must read the full file, not the green subset")
        self.assertEqual(out["11-0000.00"][0]["task"], "real task")

    def test_reworded_task_counts_as_surviving(self):
        from onet_scraper.churn import diff, normalise
        before = {"A": [{"id": "1", "task": "Analyze data.", "key": normalise("Analyze data.")}]}
        after = {"A": [{"id": "9", "task": "Analyze  data!", "key": normalise("Analyze  data!")}]}
        d = diff(before, after, ["A"])
        self.assertEqual(d["added"], 0, "punctuation and spacing are not a new task")
        self.assertEqual(d["survived"], 1)

    def test_missing_occupation_is_skipped_not_counted_as_churn(self):
        from onet_scraper.churn import diff, normalise
        before = {"A": [{"id": "1", "task": "x", "key": "x"}]}
        after = {}
        d = diff(before, after, ["A"])
        self.assertEqual(d["occupations_skipped"], 1)
        self.assertEqual(d["retired"], 0, "a taxonomy change is not a retired task")


class TestScenarios(unittest.TestCase):
    def _t(self, e, a, code="15-0001.00"):
        return {"onet_soc_code": code, "occupation_title": "X", "task_id": 1,
                "task": "t", "importance": 50, "exposure": e, "anchoring": a}

    def test_accountability_is_what_the_scenario_moves(self):
        """Capability is fixed; the scenarios differ in how much accountability
        they are willing to hand over, so a high-exposure high-anchoring task
        should only automate at the extreme setting."""
        from onet_scraper.scenarios import fate
        t = self._t(75, 60)
        self.assertEqual(fate(t, "modest"), "augmented")
        self.assertEqual(fate(t, "substantial"), "augmented")
        self.assertEqual(fate(t, "extreme"), "automated")

    def test_low_exposure_never_moves(self):
        from onet_scraper.scenarios import SCENARIOS, fate
        t = self._t(20, 20)
        for name in SCENARIOS:
            self.assertEqual(fate(t, name), "unchanged", name)

    def test_scenarios_are_ordered_by_severity(self):
        from onet_scraper.scenarios import SCENARIOS, task_fates
        import collections
        tasks = [self._t(e, a) for e in range(30, 100, 5) for a in range(20, 80, 10)]
        counts = {}
        for name in SCENARIOS:
            f = task_fates(tasks)
            counts[name] = sum(1 for r in f if r[name] == "automated")
        self.assertLess(counts["modest"], counts["substantial"])
        self.assertLess(counts["substantial"], counts["extreme"])

    def test_employment_is_counted_once_per_soc(self):
        """Several O*NET occupations share one SOC and carry the same employment
        figure; summing across the O*NET rows turns 21.5M into 49.3M."""
        from onet_scraper.scenarios import by_occupation, flows
        tasks = [self._t(90, 10, "29-1141.01"), self._t(90, 10, "29-1141.02")]
        fates = __import__("onet_scraper.scenarios", fromlist=["x"]).task_fates(tasks)
        occ = by_occupation(fates, [], {"29-1141.01": "A", "29-1141.02": "B"},
                           {"29-1141.01": 3_000_000.0, "29-1141.02": 3_000_000.0},
                           set())
        naive = flows(occ)                       # no SOC map: double counts
        correct = flows(occ, {"29-1141.01": "29-1141", "29-1141.02": "29-1141"})
        self.assertEqual(naive["substantial"]["workers"], 6_000_000)
        self.assertEqual(correct["substantial"]["workers"], 3_000_000)
        self.assertEqual(correct["substantial"]["soc_codes"], 1)

    def test_new_tasks_are_observed_not_derived(self):
        from onet_scraper.scenarios import by_occupation, task_fates
        occ = by_occupation(task_fates([self._t(50, 50)]),
                            [{"onet_soc_code": "15-0001.00", "task": "new thing"}],
                            {"15-0001.00": "X"}, {}, set())
        self.assertTrue(all(r["new_tasks"] == 1 for r in occ))


class TestSecuritySeverity(unittest.TestCase):
    def test_every_octant_has_a_severity(self):
        names = {n for n, _ in security.OCTANTS.values()}
        self.assertEqual(names, set(security.SEVERITY))

    def test_severity_tiers_are_in_range(self):
        self.assertTrue(all(0 <= v < len(security.SEVERITY_LABELS)
                            for v in security.SEVERITY.values()))
        self.assertEqual(len(security.SEVERITY_LABELS), 4)

    def test_strategic_trap_is_the_only_critical_cell(self):
        top = max(security.SEVERITY.values())
        critical = [n for n, v in security.SEVERITY.items() if v == top]
        self.assertEqual(critical, ["Strategic trap"])

    def test_irreversible_outranks_reversible_at_equal_risk(self):
        """Quiet attrition and Hold the line both sit at low efficiency; the
        first is irreversible, the second is not, and that must show up."""
        self.assertGreater(security.severity("Quiet attrition"),
                           security.severity("Hold the line"))

    def test_unknown_octant_is_lowest_not_an_error(self):
        self.assertEqual(security.severity("nonsense"), 0)


class TestSecurityEmploymentPartition(unittest.TestCase):
    """Employment must partition exactly, however the matrix is sliced."""

    SOC_OF = {"19-1029.00": "19-1029", "19-1029.01": "19-1029",
              "19-1029.02": "19-1029", "29-1141.00": "29-1141"}
    EMP = {"19-1029.00": 55_850.0, "19-1029.01": 55_850.0,
           "19-1029.02": 55_850.0, "29-1141.00": 3_000_000.0}

    def test_share_splits_a_soc_evenly_and_preserves_the_total(self):
        shares = security.employment_shares(
            list(self.SOC_OF), self.EMP, self.SOC_OF)
        self.assertAlmostEqual(shares["19-1029.00"], 55_850.0 / 3, places=4)
        self.assertEqual(shares["29-1141.00"], 3_000_000.0)
        self.assertAlmostEqual(sum(shares.values()), 55_850.0 + 3_000_000.0, places=3)

    def test_naive_soc_dedup_overcounts_a_straddling_soc(self):
        """The bug this exists to prevent: three occupations of one SOC landing
        in three different cells, each cell claiming the whole SOC."""
        shares = security.employment_shares(
            list(self.SOC_OF), self.EMP, self.SOC_OF)
        cells = [["19-1029.00"], ["19-1029.01"], ["19-1029.02"]]
        by_share = sum(sum(shares[c] for c in cell) for cell in cells)
        by_soc_dedup = sum(self.EMP[cell[0]] for cell in cells)
        self.assertAlmostEqual(by_share, 55_850.0, places=3)
        self.assertEqual(by_soc_dedup, 55_850.0 * 3)
        self.assertNotEqual(round(by_share), by_soc_dedup)

    def test_occupation_with_no_employment_gets_no_share(self):
        shares = security.employment_shares(["X.00"], {}, {"X.00": "X"})
        self.assertNotIn("X.00", shares)

    def test_cell_shares_sum_to_one(self):
        fates, scores, occs, emp, soc_of, fields = \
            TestSecurityMatrix._corpus(TestSecurityMatrix())
        rows = security.build(fates, scores, occs, emp, fields=fields, soc_of=soc_of)
        report = security.summarise(rows, soc_of)
        for scenario, s in report["scenarios"].items():
            shares = [c["share_employment"] for c in s["by_octant"].values()]
            self.assertAlmostEqual(sum(shares), 1.0, places=3, msg=scenario)
            self.assertEqual(s["total_employment"], 3_000_000.0)

    def test_field_totals_sum_to_the_corpus_total(self):
        fates, scores, occs, emp, soc_of, fields = \
            TestSecurityMatrix._corpus(TestSecurityMatrix())
        rows = security.build(fates, scores, occs, emp, fields=fields, soc_of=soc_of)
        for scenario in ("modest", "substantial", "extreme"):
            fl = [f for f in security.by_field(rows, soc_of)
                  if f["scenario"] == scenario]
            self.assertAlmostEqual(sum(f["total_employment"] for f in fl),
                                   3_000_000.0, places=3)


class TestSharedTheme(unittest.TestCase):
    """Every emitted page draws type and colour from one place.

    The dashboard and the security matrix originally shipped their own
    `-apple-system` stack and their own surface colours, which is what made the
    site read as assembled rather than designed.
    """

    def _heads(self):
        from onet_scraper import dashboard, matrix3d, theme
        return {
            "dashboard": dashboard.TEMPLATE_HEAD,
            "matrix": theme.head("t", matrix3d.CSS, root_class="m3-root",
                                 extra_tokens=matrix3d.SEV_TOKENS,
                                 extra_tokens_dark=matrix3d.SEV_TOKENS_DARK),
        }

    def test_every_page_loads_the_same_three_families(self):
        for name, head in self._heads().items():
            for family in ("Manrope", "DM Serif Display", "IBM Plex Mono"):
                self.assertIn(family, head, f"{name} is missing {family}")

    def test_no_page_falls_back_to_a_system_ui_stack(self):
        for name, head in self._heads().items():
            self.assertNotIn("-apple-system", head, name)
            self.assertNotIn("BlinkMacSystemFont", head, name)

    def test_pages_share_the_paper_and_ink_surfaces(self):
        from onet_scraper import theme
        for name, head in self._heads().items():
            self.assertIn("#faf9f5", head, f"{name} lost the paper surface")
            self.assertIn("#1c1c18", head, f"{name} lost the ink colour")
        # the near-white surface the two pages used before
        for name, head in self._heads().items():
            self.assertNotIn("#fcfcfb", head, name)

    def test_uppercase_labels_carry_tracking(self):
        """Uppercase set at its natural letter-spacing is the defect the user
        saw as bad kerning; the label class must declare tracking."""
        from onet_scraper import theme
        self.assertIn("text-transform: uppercase", theme.BASE)
        self.assertRegex(theme.BASE, r"letter-spacing:\s*\.0[6-9]em|letter-spacing:\s*\.1em")

    def test_figures_use_tabular_numerals(self):
        from onet_scraper import theme
        self.assertIn("font-variant-numeric: tabular-nums", theme.BASE)

    def test_display_serif_is_not_negatively_tracked(self):
        """DM Serif Display is already tightly fitted; the dashboard's
        -0.01/-0.02em collided the terminals at display sizes."""
        from onet_scraper import theme
        self.assertNotIn("letter-spacing: -0.0", theme.BASE)
        self.assertNotIn("letter-spacing:-0.0", theme.BASE)

    def test_extra_tokens_reach_all_three_selectors(self):
        from onet_scraper import theme
        head = theme.head("t", extra_tokens="  --x: red;\n",
                          extra_tokens_dark="  --x: blue;\n")
        self.assertEqual(head.count("--x: red"), 1)      # light root
        self.assertEqual(head.count("--x: blue"), 2)     # media query + data-theme

    def test_dashboard_reads_data_not_window_data(self):
        """const at script scope does not become a window property, so
        window.DATA left the handoff columns blank."""
        from onet_scraper import dashboard
        import inspect
        src = inspect.getsource(dashboard)
        self.assertNotIn("window.DATA?.hand", src)
        self.assertIn("new Map((DATA.hand || [])", src)


class TestSecurityWeighting(unittest.TestCase):
    """No task may be silently dropped from its own occupation's score."""

    def _t(self, **kw):
        base = {"importance": 50.0, "substantial": "unchanged", "error_cost": 50.0,
                "accountability_requirement": 50.0, "judgment_under_uncertainty": 50.0}
        base.update(kw)
        return base

    def test_unrated_task_takes_the_occupations_mean(self):
        tasks = [self._t(importance=80.0), self._t(importance=40.0),
                 self._t(importance="")]
        self.assertEqual(security.weights(tasks), [80.0, 40.0, 60.0])

    def test_wholly_unrated_occupation_falls_back_to_equal_weights(self):
        tasks = [self._t(importance=""), self._t(importance=""), self._t(importance="")]
        self.assertEqual(security.weights(tasks), [1.0, 1.0, 1.0])

    def test_a_surgeon_with_no_importance_ratings_is_not_scored_zero(self):
        """The bug: O*NET rates no task importance for Cardiologists, Pediatric
        and Orthopedic Surgeons or EMTs, the weighted mean divided by zero, and
        they were returned as risk 0.0 and filed under 'Quiet attrition'."""
        tasks = [self._t(importance="", error_cost=95.0,
                         accountability_requirement=95.0,
                         judgment_under_uncertainty=90.0) for _ in range(5)]
        risk = security.removal_risk(tasks)
        self.assertGreater(risk, 80.0)
        self.assertNotEqual(risk, 0.0)
        name, _ = security.octant(20.0, risk, 85.0)
        self.assertEqual(name, "Protect")

    def test_efficiency_is_not_zero_for_a_wholly_unrated_occupation(self):
        tasks = [self._t(importance="", substantial="automated") for _ in range(4)]
        self.assertEqual(security.efficiency(tasks, "substantial"), 100.0)

    def test_partially_rated_occupation_still_counts_every_task(self):
        """Radiologists had 13 of 30 tasks unrated; dropping them changed the
        occupation's cell, so the unrated ones have to carry weight."""
        rated_only = [self._t(importance=90.0, substantial="unchanged")]
        with_unrated = rated_only + [self._t(importance="", substantial="automated")]
        self.assertEqual(security.efficiency(rated_only, "substantial"), 0.0)
        self.assertGreater(security.efficiency(with_unrated, "substantial"), 0.0)

    def test_weights_length_always_matches_task_count(self):
        for n in (0, 1, 5):
            tasks = [self._t() for _ in range(n)]
            self.assertEqual(len(security.weights(tasks)), n)

    def test_negative_and_zero_importance_are_treated_as_missing(self):
        """O*NET's sort sentinel puts -2.0 behind 'Not available'."""
        tasks = [self._t(importance=60.0), self._t(importance=-2.0),
                 self._t(importance=0.0)]
        self.assertEqual(security.weights(tasks), [60.0, 60.0, 60.0])


class TestDerivedValidation(unittest.TestCase):
    """Each check is proved against the bug it was written for."""

    def _corpus(self, **over):
        """A minimal passing corpus, so a test can break exactly one thing."""
        def row(title, scenario, eff, risk, recon=70.0, code=None):
            return {"onet_soc_code": code or title[:6], "title": title,
                    "scenario": scenario, "efficiency": eff, "removal_risk": risk,
                    "reconstitution": recon, "trap_score": 10.0}
        security = []
        for title, risk in (("Pediatric Surgeons", 78.0), ("Cardiologists", 79.0),
                            ("Anesthesiologists", 76.0),
                            ("Emergency Medical Technicians", 75.0),
                            ("Oral and Maxillofacial Surgeons", 88.0),
                            ("Video Game Designers", 36.0),
                            ("Business Intelligence Analysts", 35.0)):
            for sc, eff in (("modest", 30.0), ("substantial", 50.0), ("extreme", 70.0)):
                security.append(row(title, sc, eff, risk))
        susc = [{"title": "Business Intelligence Analysts", "exposure": 86.0,
                 "susceptibility": 80.0, "anchoring": 26.0},
                {"title": "Mathematicians", "exposure": 84.0,
                 "susceptibility": 77.0, "anchoring": 30.0},
                {"title": "Anesthesiologists", "exposure": 40.0,
                 "susceptibility": 30.0, "anchoring": 70.0},
                {"title": "Paramedics", "exposure": 38.0,
                 "susceptibility": 28.0, "anchoring": 72.0}]
        soc = [{"total_employment": 1000.0}]
        report = {"scenarios": {sc: {"total_employment": 1000.0, "by_octant": {
            "Strategic trap": {"share_employment": 0.5},
            "Protect": {"share_employment": 0.5}}}
            for sc in ("modest", "substantial", "extreme")}}
        data = {"susceptibility": susc, "handoff": [], "security": security,
                "security_report": report, "soc": soc}
        data.update(over)
        return data

    def _run(self, **over):
        from onet_scraper.validate_derived import validate_derived
        checks, summary = validate_derived(**self._corpus(**over))
        return {c["check"]: c for c in checks}, summary

    def test_the_baseline_corpus_passes(self):
        checks, summary = self._run()
        failed = [n for n, c in checks.items() if not c["passed"]]
        self.assertEqual(failed, [], f"baseline should pass, failed: {failed}")
        self.assertEqual(summary["errors"], 0)

    def test_catches_a_composite_of_exactly_zero(self):
        """The surgeons bug: no importance ratings, empty denominator, 0.0."""
        c = self._corpus()
        for r in c["security"]:
            if r["title"] == "Pediatric Surgeons":
                r["removal_risk"] = 0.0
        checks, summary = self._run(security=c["security"])
        self.assertFalse(checks["no_composite_is_exactly_zero"]["passed"])
        self.assertFalse(checks["removal_risk_anchors_hold"]["passed"])
        self.assertGreater(summary["errors"], 0)

    def test_catches_an_out_of_range_index(self):
        c = self._corpus()
        c["security"][0]["efficiency"] = 140.0
        checks, _ = self._run(security=c["security"])
        self.assertFalse(checks["indices_in_range"]["passed"])

    def test_catches_shares_that_do_not_partition(self):
        """The employment double-count: a SOC credited to several cells made the
        shares sum to 1.30."""
        c = self._corpus()
        c["security_report"]["scenarios"]["substantial"]["by_octant"][
            "Protect"]["share_employment"] = 0.8
        checks, _ = self._run(security_report=c["security_report"])
        self.assertFalse(checks["cell_shares_partition"]["passed"])

    def test_catches_employment_diverging_from_the_soc_rollup(self):
        checks, _ = self._run(soc=[{"total_employment": 49_300_000.0}])
        self.assertFalse(checks["employment_matches_soc_rollup"]["passed"])

    def test_catches_non_monotonic_scenarios(self):
        """A threshold edit applied to the wrong scenario."""
        c = self._corpus()
        for r in c["security"]:
            if r["scenario"] == "extreme":
                r["efficiency"] = 5.0
        checks, _ = self._run(security=c["security"])
        self.assertFalse(checks["efficiency_rises_with_scenario"]["passed"])

    def test_catches_a_missing_axis(self):
        c = self._corpus()
        c["security"][1]["reconstitution"] = ""
        checks, _ = self._run(security=c["security"])
        self.assertFalse(checks["every_occupation_has_all_three_axes"]["passed"])

    def test_anchor_outside_its_band_fails_with_a_reason(self):
        """A failure has to say why the band exists, or whoever hits it cannot
        judge whether the anchor or the change is wrong."""
        c = self._corpus()
        for r in c["security"]:
            if r["title"] == "Video Game Designers":
                r["removal_risk"] = 95.0
        checks, _ = self._run(security=c["security"])
        bad = checks["removal_risk_anchors_hold"]
        self.assertFalse(bad["passed"])
        self.assertTrue(any("because" in s for s in bad["sample"]))

    def test_a_missing_anchor_warns_rather_than_failing(self):
        """An anchor that is not in the corpus is a dead test, and has to say
        so - but it must not fail a build that is otherwise sound."""
        checks, summary = self._run(security=[
            r for r in self._corpus()["security"] if r["title"] != "Cardiologists"])
        self.assertFalse(checks["removal_risk_anchors_present"]["passed"])
        self.assertEqual(checks["removal_risk_anchors_present"]["severity"], "warn")
        self.assertEqual(summary["errors"], 0)

    def test_every_declared_anchor_resolves_against_the_real_corpus(self):
        """Guards against an anchor quietly going stale when O*NET renames an
        occupation - which has already happened to three work_context elements."""
        import csv
        from pathlib import Path
        from onet_scraper.validate_derived import EXPOSURE_ANCHORS, RISK_ANCHORS
        path = Path("data/out/security_matrix.csv")
        if not path.exists():
            self.skipTest("no built dataset")
        titles = [r["title"] for r in csv.DictReader(path.open())]
        for needle, *_ in RISK_ANCHORS + EXPOSURE_ANCHORS:
            self.assertTrue(any(needle.lower() in t.lower() for t in titles),
                            f"anchor {needle!r} no longer matches any occupation")


class TestOnetRatings(unittest.TestCase):
    """Recurrence, precision and education depth from O*NET's own scales."""

    def _ft(self, code, task, dist):
        return [{"O*NET-SOC Code": code, "Task ID": task, "Scale ID": "FT",
                 "Category": str(c), "Data Value": str(v), "N": "40",
                 "Standard Error": "1.0"} for c, v in dist.items()]

    def test_recurrence_is_log_scaled_not_a_category_average(self):
        """FT bands are roughly log-spaced in frequency. Averaging the band
        numbers would make monthly-to-weekly the same step as daily-to-hourly,
        which is wrong by more than an order of magnitude in the thing being
        measured."""
        from onet_scraper.onet_ratings import recurrence
        # Band 4 is "more than weekly", ~100 times a year. On the category
        # numbers it is the midpoint of 1-7, so a linear scale puts it at 50.
        # On frequency it is 100 of a possible 2000, and the log scale puts it
        # at 61 - the bands are not evenly spaced in the thing they measure.
        got = recurrence(self._ft("X", "1", {4: 100.0}))[("X", "1")]
        self.assertAlmostEqual(got, 60.6, places=0)
        self.assertGreater(got, 55.0)
        # a symmetric split of the two endpoints does land on the midpoint, by
        # construction of the rescaling - that is not evidence either way
        ends = recurrence(self._ft("X", "2", {1: 50.0, 7: 50.0}))[("X", "2")]
        self.assertAlmostEqual(ends, 50.0, places=1)

    def test_recurrence_endpoints_anchor_the_scale(self):
        from onet_scraper.onet_ratings import recurrence
        self.assertEqual(recurrence(self._ft("X", "1", {1: 100.0}))[("X", "1")], 0.0)
        self.assertEqual(recurrence(self._ft("X", "2", {7: 100.0}))[("X", "2")], 100.0)

    def test_recurrence_ranking_survives_the_rate_estimates(self):
        """The per-year rates are ours, not O*NET's - O*NET names the bands. The
        ordering must not depend on the exact numbers."""
        from onet_scraper import onet_ratings as R
        rare = self._ft("A", "1", {2: 100.0})
        often = self._ft("B", "1", {6: 100.0})
        original = dict(R.FT_PER_YEAR)
        try:
            for scale in (0.5, 1.0, 3.0):
                R.FT_PER_YEAR = {k: v * scale for k, v in original.items()}
                self.assertLess(R.recurrence(rare)[("A", "1")],
                                R.recurrence(often)[("B", "1")])
        finally:
            R.FT_PER_YEAR = original

    def test_occupation_recurrence_is_importance_weighted(self):
        from onet_scraper.onet_ratings import occupation_recurrence
        task = {("X", "1"): 20.0, ("X", "2"): 80.0}
        even = occupation_recurrence(task)["X"]
        weighted = occupation_recurrence(task, {("X", "1"): 90.0, ("X", "2"): 10.0})["X"]
        self.assertAlmostEqual(even, 50.0, places=1)
        self.assertLess(weighted, even)   # the rare task now dominates

    def test_education_depth_uses_years_not_category_numbers(self):
        """RL categories are unevenly spaced in time: associate's to bachelor's
        is two years, master's to post-master's certificate is one."""
        from onet_scraper.onet_ratings import education_depth
        rows = [{"O*NET-SOC Code": "X", "Scale ID": "RL", "Category": "6",
                 "Data Value": "100.0"}]
        got = education_depth(rows)["X"]
        self.assertEqual(got["years"], 16.0)          # bachelor's
        self.assertGreater(got["depth"], 40.0)

    def test_education_depth_averages_the_distribution(self):
        from onet_scraper.onet_ratings import education_depth
        rows = [{"O*NET-SOC Code": "X", "Scale ID": "RL", "Category": "2",
                 "Data Value": "50.0"},
                {"O*NET-SOC Code": "X", "Scale ID": "RL", "Category": "11",
                 "Data Value": "50.0"}]
        self.assertAlmostEqual(education_depth(rows)["X"]["years"], 16.5, places=2)

    def test_education_beats_job_zone_on_resolution(self):
        """Job Zone gave three distinct values across this corpus and assigned
        an ophthalmic technician the same training depth as a
        neuropsychologist."""
        from onet_scraper.onet_ratings import EDUCATION_YEARS, education_depth
        self.assertGreaterEqual(len(EDUCATION_YEARS), 12)
        rows = []
        for i, cat in enumerate(range(1, 13)):
            rows.append({"O*NET-SOC Code": f"X{i}", "Scale ID": "RL",
                         "Category": str(cat), "Data Value": "100.0"})
        depths = {v["depth"] for v in education_depth(rows).values()}
        self.assertGreaterEqual(len(depths), 10)

    def test_reconstitution_prefers_education_and_falls_back_to_job_zone(self):
        from onet_scraper.security import reconstitution
        with_ed = reconstitution(3, 10_000, 0.3, 3.0, 6.0, education_depth=90.0)
        fallback = reconstitution(3, 10_000, 0.3, 3.0, 6.0)
        self.assertEqual(with_ed["depth_source"], "education")
        self.assertEqual(fallback["depth_source"], "job_zone")
        self.assertEqual(with_ed["training_depth"], 90.0)
        self.assertGreater(with_ed["reconstitution"], fallback["reconstitution"])

    def test_precision_reports_the_median_respondent_count(self):
        from onet_scraper.onet_ratings import rating_precision
        rows = [{"O*NET-SOC Code": "X", "Scale ID": "IM", "N": n,
                 "Standard Error": "1.0"} for n in ("4", "40", "200")]
        got = rating_precision(rows)["X"]
        self.assertEqual(got["respondents"], 40.0)
        self.assertEqual(got["respondents_min"], 4.0)

    def test_thin_samples_are_flagged(self):
        from onet_scraper.onet_ratings import rating_precision
        rows = [{"O*NET-SOC Code": "X", "Scale ID": "IM", "N": "5",
                 "Standard Error": "2.0"}]
        self.assertEqual(rating_precision(rows)["X"]["thin_sample"], 1)

    def test_other_scales_are_ignored(self):
        """task_ratings.csv carries IM, RT and FT in one file."""
        from onet_scraper.onet_ratings import recurrence
        rows = self._ft("X", "1", {5: 100.0})
        rows.append({"O*NET-SOC Code": "X", "Task ID": "1", "Scale ID": "IM",
                     "Category": "", "Data Value": "77.0", "N": "40"})
        self.assertEqual(len(recurrence(rows)), 1)

    def test_recurrence_opposes_the_existing_tractability_terms(self):
        """The reason it is measured but not folded in. Repetitive STEM work is
        hands-on work: recurrence runs against llm_exposure and with physical
        embodiment, so averaging it in cancels signal rather than adding it."""
        import csv
        import statistics
        from pathlib import Path
        path = Path("data/out/occupation_handoff.csv")
        if not path.exists():
            self.skipTest("no built dataset")
            return
        rows = [r for r in csv.DictReader(path.open()) if r.get("recurrence")]
        if len(rows) < 50:
            self.skipTest("recurrence not populated; run fetch-bulk --with-ratings")
            return
        t = [float(r["tractability"]) for r in rows]
        c = [float(r["recurrence"]) for r in rows]
        mt, mc = statistics.fmean(t), statistics.fmean(c)
        cov = sum((a - mt) * (b - mc) for a, b in zip(t, c)) / len(t)
        r = cov / (statistics.pstdev(t) * statistics.pstdev(c))
        self.assertLess(r, -0.3, f"expected a negative correlation, got r={r:.2f}")
        self.assertGreater(r, -0.8, f"r={r:.2f} would make it near-redundant")


class TestFrontierCalibration(unittest.TestCase):
    """The frontier is derived from the corpus, not asserted against it."""

    def _corpus(self, n=200, recurrence=False):
        """A synthetic corpus with a realistic spread on both axes."""
        rows = []
        for i in range(n):
            rows.append({
                "onet_soc_code": f"{i:02d}-0000.00",
                "llm_exposure": 30.0 + (i % 50),
                "automation_feasibility_today": 10.0 + (i % 30),
                "physical_embodiment_required": 5.0 + (i % 60),
                "judgment_under_uncertainty": 20.0 + (i % 45),
                "accountability_requirement": 25.0 + (i % 55),
                "error_cost": 20.0 + (i % 65),
                "interpersonal_demand": 15.0 + (i % 40),
            })
        return rows

    def _classify_all(self, rows, recurrence=None, absolute=False):
        from onet_scraper import handoff as H
        pairs = [H.axes(r, (recurrence or {}).get(r["onet_soc_code"])) for r in rows]
        gaps = [float(r["llm_exposure"]) - float(r["automation_feasibility_today"])
                for r in rows]
        cal = (H.Calibration.absolute() if absolute
               else H.calibrate([t for t, _ in pairs], [x for _, x in pairs], gaps))
        return ({r["onet_soc_code"]: H.classify(t, x, g, cal)
                 for r, (t, x), g in zip(rows, pairs, gaps)}, cal)

    # -- the identity that makes this a reparameterisation ----------------
    def test_derived_calibration_reproduces_the_absolutes_on_the_reference(self):
        """The quantiles were inverted against the single-model pass, so they
        reproduce the hand-chosen constants on THAT distribution.

        They deliberately do not reproduce them on the live corpus any more:
        consolidating two scoring passes moved the axes, and the thresholds
        moved with them - frontier_k from 53.0 to 51.1, the watch-point
        willingness gap from 40 to 35.4. That adaptation is the entire purpose
        of expressing them as quantiles, so asserting identity against whatever
        corpus happens to be current would assert the opposite of the design.
        """
        import csv
        import json
        from pathlib import Path
        from onet_scraper import handoff as H
        from onet_scraper.score import propagate
        ckpt = Path("data/raw/subtask_scores.jsonl")
        if not ckpt.exists() or not Path("data/out/tasks.csv").exists():
            self.skipTest("no reference checkpoint")
            return
        rows = [json.loads(l) for l in ckpt.read_text().splitlines() if l.strip()]
        rt = lambda n: list(csv.DictReader(open(f"data/out/{n}.csv")))
        _, scored = propagate(rows, rt("tasks"), rt("task_subtasks"), rt("occupations"))
        pairs = [H.axes(r) for r in scored]
        cal = H.calibrate(
            [t for t, _ in pairs], [x for _, x in pairs],
            [float(r["llm_exposure"]) - float(r["automation_feasibility_today"])
             for r in scored])
        for field, original in (("tractability_floor", 50.0), ("frontier_k", 53.0),
                                ("crossing_band", 5.0), ("watch_tractability", 55.0),
                                ("watch_resistance", 50.0), ("watch_gap", 40.0)):
            self.assertAlmostEqual(getattr(cal, field), original, places=2,
                                   msg=f"{field} drifted from its absolute")

    def test_reparameterisation_changes_no_classification(self):
        """Architects sit 0.0007 from the crossing boundary, so this is a real
        constraint on the precision of CROSSING_BAND_SD, not a formality."""
        import csv
        from pathlib import Path
        from onet_scraper import handoff as H
        if not Path("data/out/occupation_handoff.csv").exists():
            self.skipTest("no built dataset")
            return
        published = {r["onet_soc_code"]: r["classification"] for r in
                     csv.DictReader(open("data/out/occupation_handoff.csv"))}
        scored = list(csv.DictReader(
            open("data/out/occupation_automation_scores.csv")))
        got = {r["onet_soc_code"]: r["classification"] for r in H.build(scored)}
        # Most classifications must survive a rescoring: the axes moved when the
        # two passes were consolidated, so a handful legitimately changed, but a
        # wholesale reshuffle would mean the calibration is not tracking.
        changed = [c for c in published if published[c] != got.get(c)]
        self.assertLess(len(changed) / max(len(published), 1), 0.12,
                        f"{len(changed)} of {len(published)} classifications moved")

    # -- the fragility it removes ----------------------------------------
    def test_calibration_tracks_a_compressed_axis(self):
        rows = self._corpus()
        rec = {r["onet_soc_code"]: 50.0 + (i % 40)
               for i, r in enumerate(rows)}
        _, three = self._classify_all(rows)
        _, four = self._classify_all(rows, recurrence=rec)
        # adding a fourth term narrows the axis, so the thresholds must move
        self.assertNotAlmostEqual(three.tractability_floor, four.tractability_floor,
                                  places=2)

    def test_stale_absolutes_and_adaptive_disagree_on_a_changed_axis(self):
        """The whole point. On the real corpus, adding recurrence under the old
        fixed constants raised "handed off" from 36 to 60; under a calibration
        that tracks the axis it falls to 25. The stale thresholds inverted the
        direction of the conclusion, so anything that reintroduces fixed
        constants has to fail here."""
        rows = self._corpus()
        rec = {r["onet_soc_code"]: 50.0 + (i % 40) for i, r in enumerate(rows)}
        stale, _ = self._classify_all(rows, recurrence=rec, absolute=True)
        adaptive, _ = self._classify_all(rows, recurrence=rec)
        disagree = [c for c in stale if stale[c] != adaptive[c]]
        self.assertGreater(len(disagree), 0,
                           "a compressed axis must classify differently under "
                           "fixed thresholds than under derived ones")

    def test_an_empty_corpus_falls_back_to_the_absolutes(self):
        from onet_scraper.handoff import Calibration, calibrate
        self.assertEqual(calibrate([], [], []), Calibration.absolute())

    def test_classify_without_a_calibration_uses_the_absolutes(self):
        from onet_scraper import handoff as H
        # tractability below the absolute floor is human-held either way
        self.assertEqual(H.classify(40.0, 60.0, 10.0), "Human held")

    def test_quantile_interpolates(self):
        from onet_scraper.handoff import _quantile
        v = [0.0, 10.0, 20.0, 30.0]
        self.assertEqual(_quantile(v, 0.0), 0.0)
        self.assertEqual(_quantile(v, 1.0), 30.0)
        self.assertAlmostEqual(_quantile(v, 0.5), 15.0)
        self.assertEqual(_quantile([], 0.5), 0.0)

    def test_frontier_is_a_threshold_on_the_product(self):
        """k is calibrated on T*R because the curve is the locus T*R = k^2."""
        from onet_scraper.handoff import calibrate
        t = [40.0, 50.0, 60.0, 70.0]
        r = [70.0, 60.0, 50.0, 40.0]   # every product is near 2800-3000
        cal = calibrate(t, r, [10.0])
        self.assertGreater(cal.frontier_k, 45.0)
        self.assertLess(cal.frontier_k, 60.0)


class TestReliability(unittest.TestCase):
    """Agreement between two scoring passes."""

    def _rows(self, vals, key="dwa_id"):
        from onet_scraper.reliability import DIMENSIONS
        return [{key: f"d{i}", **{d: v for d in DIMENSIONS}}
                for i, v in enumerate(vals)]

    def test_identical_passes_agree_perfectly(self):
        from onet_scraper.reliability import compare
        vals = [10, 25, 40, 55, 70, 85, 95]
        rep = compare(self._rows(vals), self._rows(vals))
        self.assertEqual(rep["subtasks_compared"], 7)
        self.assertAlmostEqual(rep["mean_pearson"], 1.0, places=4)
        self.assertAlmostEqual(rep["mean_icc"], 1.0, places=4)
        self.assertEqual(rep["mean_abs_diff"], 0.0)

    def test_icc_punishes_a_constant_offset_where_r_does_not(self):
        """Two raters who disagree by a flat 20 points correlate at 1.0. For
        scores meant to be interchangeable that is the wrong answer, which is
        why ICC is reported alongside."""
        from onet_scraper.reliability import compare
        vals = [10, 25, 40, 55, 70, 85, 95]
        rep = compare(self._rows(vals), self._rows([v + 20 for v in vals]))
        self.assertAlmostEqual(rep["mean_pearson"], 1.0, places=4)
        # The claim is the relationship, not a threshold: how far ICC falls
        # depends on the offset relative to the spread of the targets, so a
        # fixed cutoff is an arbitrary one. Here it lands at 0.907; on a
        # narrower spread the same offset gives 0.824.
        self.assertLess(rep["mean_icc"], rep["mean_pearson"] - 0.05)
        self.assertEqual(rep["mean_abs_diff"], 20.0)

        # and the penalty grows as the offset grows
        worse = compare(self._rows(vals), self._rows([v + 45 for v in vals]))
        self.assertLess(worse["mean_icc"], rep["mean_icc"])

    def test_signed_difference_separates_bias_from_noise(self):
        from onet_scraper.reliability import compare
        vals = [10, 25, 40, 55, 70, 85, 95]
        rep = compare(self._rows(vals), self._rows([v + 12 for v in vals]))
        d = rep["dimensions"]["llm_exposure"]
        self.assertAlmostEqual(d["mean_signed_diff"], 12.0, places=1)
        self.assertAlmostEqual(d["mean_abs_diff"], 12.0, places=1)

    def test_no_overlap_reports_nothing_rather_than_zero_agreement(self):
        """A run that scored nothing must not surface as r = 0.000 over n = 0 -
        a failed measurement presented as a finished one."""
        from onet_scraper.reliability import compare
        rep = compare(self._rows([10, 20, 30]), [])
        self.assertEqual(rep["subtasks_compared"], 0)
        self.assertIsNone(rep["mean_pearson"])

    def test_partial_overlap_is_counted_honestly(self):
        from onet_scraper.reliability import compare
        a = self._rows([10, 20, 30, 40])
        b = self._rows([10, 20, 30, 40])[:2]
        rep = compare(a, b)
        self.assertEqual(rep["subtasks_compared"], 2)
        self.assertEqual(rep["only_in_first"], 2)

    def test_kind_is_recorded_so_the_report_cannot_mislabel_itself(self):
        """test-retest and cross-model do not license the same claim."""
        from onet_scraper.reliability import compare
        vals = [10, 30, 50, 70]
        self.assertEqual(compare(self._rows(vals), self._rows(vals),
                                 kind="test-retest")["kind"], "test-retest")
        self.assertEqual(compare(self._rows(vals), self._rows(vals))["kind"],
                         "cross-model")

    def test_a_flat_rater_yields_no_correlation_rather_than_a_crash(self):
        from onet_scraper.reliability import compare
        rep = compare(self._rows([10, 20, 30, 40]), self._rows([50, 50, 50, 50]))
        self.assertIsNone(rep["dimensions"]["llm_exposure"]["pearson"])

    def test_within_bands_are_reported(self):
        from onet_scraper.reliability import compare
        a = self._rows([10, 20, 30, 40, 50])
        b = self._rows([15, 28, 30, 65, 52])
        d = compare(a, b)["dimensions"]["error_cost"]
        self.assertEqual(d["within_10"], 0.8)     # four of five within 10
        self.assertEqual(d["within_20"], 0.8)


class TestConsolidation(unittest.TestCase):
    """Merging two scoring passes into one canonical set."""

    def _row(self, i, **over):
        from onet_scraper.reliability import DIMENSIONS
        r = {"dwa_id": f"d{i}", "dwa_title": f"t{i}", "model": "m1"}
        r.update({d: 50.0 for d in DIMENSIONS})
        r.update(over)
        return r

    def test_two_raters_are_averaged(self):
        from onet_scraper.reliability import consolidate
        a = [self._row(1, llm_exposure=80.0, model="opus")]
        b = [self._row(1, llm_exposure=60.0, model="sonnet")]
        out = consolidate(a, b)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["llm_exposure"], 70.0)
        self.assertEqual(out[0]["n_raters"], 2)
        self.assertEqual(out[0]["raters"], "opus;sonnet")

    def test_disagreement_is_recorded_as_the_uncertainty(self):
        from onet_scraper.reliability import consolidate
        a = [self._row(1, llm_exposure=80.0, error_cost=40.0)]
        b = [self._row(1, llm_exposure=60.0, error_cost=50.0)]
        out = consolidate(a, b)
        # two dimensions differ by 20 and 10, the other five by 0. The stored
        # value is rounded to one decimal, which is the resolution the scores
        # themselves have.
        self.assertAlmostEqual(out[0]["score_disagreement"], 30 / 7, places=1)

    def test_a_subtask_only_one_rater_reached_keeps_its_score(self):
        """Null disagreement, not zero - an unmeasured spread must not read as
        perfect agreement."""
        from onet_scraper.reliability import consolidate
        out = consolidate([self._row(1, llm_exposure=80.0)], [])
        self.assertEqual(out[0]["llm_exposure"], 80.0)
        self.assertEqual(out[0]["n_raters"], 1)
        self.assertIsNone(out[0]["score_disagreement"])

    def test_the_union_is_kept_not_the_intersection(self):
        from onet_scraper.reliability import consolidate
        out = consolidate([self._row(1)], [self._row(2)])
        self.assertEqual(len(out), 2)
        self.assertEqual({r["n_raters"] for r in out}, {1})

    def test_summary_reports_the_tail(self):
        from onet_scraper.reliability import consolidate, consolidation_summary
        a = [self._row(i, llm_exposure=50.0) for i in range(10)]
        b = [self._row(i, llm_exposure=50.0) for i in range(10)]
        b[0]["llm_exposure"] = 100.0     # one wild disagreement
        rep = consolidation_summary(consolidate(a, b))
        # renamed from scored_by_two when consolidation went to N raters
        self.assertEqual(rep["scored_by_multiple"], 10)
        self.assertEqual(rep["by_rater_count"], {2: 10})
        self.assertEqual(rep["above_15_points"], 0)   # 50/7 = 7.1, under 15
        self.assertGreater(rep["max_disagreement"], rep["median_disagreement"])


class TestScenarioCalibration(unittest.TestCase):
    """Scenario thresholds travel with the scale."""

    def _tasks(self, exposures, anchoring=30.0):
        return [{"exposure": e, "anchoring": anchoring} for e in exposures]

    def test_quantiles_are_fixed_constants_not_recomputed_from_the_corpus(self):
        """A first version computed q(vals, pct_of(vals, 80)), which recovers 80
        by construction - circular, and a no-op on every corpus. The quantiles
        have to be constants for the cut to travel with the distribution."""
        from onet_scraper.scenarios import REFERENCE_QUANTILES, calibrate
        self.assertIn("substantial", REFERENCE_QUANTILES)
        low = calibrate(self._tasks(list(range(0, 50))))
        high = calibrate(self._tasks(list(range(50, 100))))
        self.assertLess(low["substantial"]["auto_exposure"],
                        high["substantial"]["auto_exposure"])

    def test_a_uniform_shift_in_the_scale_preserves_the_classification(self):
        """The whole purpose. Two raters who rank identically and read the scale
        ten points apart must classify the same tasks the same way."""
        from onet_scraper.scenarios import SCENARIOS, calibrate, fate
        base = list(range(10, 100, 2))
        a = self._tasks(base)
        b = self._tasks([e - 10 for e in base])
        for name in SCENARIOS:
            ca, cb = calibrate(a), calibrate(b)
            na = sum(1 for t in a if fate(t, name, ca) == "automated")
            nb = sum(1 for t in b if fate(t, name, cb) == "automated")
            self.assertEqual(na, nb, f"{name} moved under a uniform shift")

    def test_absolute_thresholds_do_not_survive_the_same_shift(self):
        """Stated as a test so nobody reverts to them: on the real corpus the
        automated count moved -50%, -37% and -28% between two scoring passes."""
        from onet_scraper.scenarios import fate
        base = list(range(10, 100, 2))
        a = self._tasks(base)
        b = self._tasks([e - 10 for e in base])
        na = sum(1 for t in a if fate(t, "substantial") == "automated")
        nb = sum(1 for t in b if fate(t, "substantial") == "automated")
        self.assertNotEqual(na, nb)

    def test_an_empty_corpus_falls_back_to_the_absolutes(self):
        from onet_scraper.scenarios import SCENARIOS, calibrate
        cal = calibrate([])
        self.assertEqual(cal["modest"]["auto_exposure"],
                         SCENARIOS["modest"]["auto_exposure"])

    def test_fate_without_a_calibration_uses_the_absolutes(self):
        from onet_scraper.scenarios import fate
        self.assertEqual(fate({"exposure": 95.0, "anchoring": 10.0}, "modest"),
                         "automated")
        self.assertEqual(fate({"exposure": 20.0, "anchoring": 10.0}, "modest"),
                         "unchanged")


class TestMultiRaterConsolidation(unittest.TestCase):
    """Consolidation over more than two passes."""

    def _row(self, i, val, model):
        from onet_scraper.reliability import DIMENSIONS
        r = {"dwa_id": f"d{i}", "model": model}
        r.update({d: val for d in DIMENSIONS})
        return r

    def test_three_passes_are_averaged(self):
        from onet_scraper.reliability import consolidate
        out = consolidate([self._row(1, 30.0, "a")],
                          [self._row(1, 60.0, "b")],
                          [self._row(1, 90.0, "c")])
        self.assertEqual(out[0]["llm_exposure"], 60.0)
        self.assertEqual(out[0]["n_raters"], 3)
        self.assertEqual(out[0]["raters"], "a;b;c")

    def test_disagreement_is_the_mean_pairwise_difference(self):
        """With three raters there are three pairs, not one difference."""
        from onet_scraper.reliability import consolidate
        out = consolidate([self._row(1, 30.0, "a")],
                          [self._row(1, 60.0, "b")],
                          [self._row(1, 90.0, "c")])
        # pairs are 30, 60, 30 -> mean 40
        self.assertAlmostEqual(out[0]["score_disagreement"], 40.0, places=1)

    def test_a_subtask_missing_from_one_pass_uses_the_others(self):
        from onet_scraper.reliability import consolidate
        out = {r["dwa_id"]: r for r in consolidate(
            [self._row(1, 40.0, "a"), self._row(2, 40.0, "a")],
            [self._row(1, 60.0, "b")],
            [self._row(1, 80.0, "c")])}
        self.assertEqual(out["d1"]["n_raters"], 3)
        self.assertEqual(out["d2"]["n_raters"], 1)
        self.assertIsNone(out["d2"]["score_disagreement"])

    def test_a_single_pass_is_a_no_op_not_an_error(self):
        from onet_scraper.reliability import consolidate
        out = consolidate([self._row(1, 50.0, "a")])
        self.assertEqual(out[0]["n_raters"], 1)
        self.assertEqual(out[0]["llm_exposure"], 50.0)

    def test_no_passes_returns_nothing(self):
        from onet_scraper.reliability import consolidate
        self.assertEqual(consolidate(), [])

    def test_two_passes_still_behave_as_before(self):
        """The migration to N raters must not move the existing result."""
        from onet_scraper.reliability import consolidate
        out = consolidate([self._row(1, 80.0, "a")], [self._row(1, 60.0, "b")])
        self.assertEqual(out[0]["llm_exposure"], 70.0)
        self.assertAlmostEqual(out[0]["score_disagreement"], 20.0, places=1)


class TestCohenKappa(unittest.TestCase):
    """Binary agreement, corrected for chance."""

    def test_perfect_agreement(self):
        from onet_scraper.reliability import cohen_kappa
        self.assertAlmostEqual(cohen_kappa([1,0,1,1,0],[1,0,1,1,0]), 1.0, places=6)

    def test_a_constant_rater_is_undefined_not_perfect(self):
        """Two raters who say yes to everything agree 100% of the time and have
        said nothing. Percent agreement would call that perfect."""
        from onet_scraper.reliability import cohen_kappa
        self.assertIsNone(cohen_kappa([1,1,1,1],[1,1,1,1]))

    def test_chance_agreement_is_discounted(self):
        """With 90% of one class, 90% raw agreement is barely better than
        chance, and kappa has to say so."""
        from onet_scraper.reliability import cohen_kappa
        k = cohen_kappa([1]*9+[0], [1]*8+[0,1])
        self.assertLess(k, 0.5)

    def test_systematic_disagreement_is_negative(self):
        from onet_scraper.reliability import cohen_kappa
        self.assertLess(cohen_kappa([1,1,0,0],[0,0,1,1]), 0)

    def test_mismatched_lengths_return_none(self):
        from onet_scraper.reliability import cohen_kappa
        self.assertIsNone(cohen_kappa([1,0],[1]))
        self.assertIsNone(cohen_kappa([],[]))


class TestUncertainty(unittest.TestCase):
    """The bootstrap, and the noise model behind it."""

    def test_sigma_is_the_difference_sd_over_root_two(self):
        """Two independent draws differ with sd sqrt(2)*sigma. Using the
        difference sd directly overstates the noise by 41%, which is one of the
        three errors in the proxy estimate this replaced."""
        import random
        from onet_scraper.uncertainty import DIMENSIONS, noise_model
        rng = random.Random(3)
        SIGMA = 6.0
        a = [{"dwa_id": f"d{i}", **{d: 50.0 for d in DIMENSIONS}} for i in range(400)]
        b = [{"dwa_id": f"d{i}",
              **{d: 50.0 + rng.gauss(0, SIGMA) - rng.gauss(0, SIGMA) if False
                 else 50.0 + rng.gauss(0, SIGMA * (2 ** 0.5)) for d in DIMENSIONS}}
             for i in range(400)]
        got = noise_model(a, b)
        for d in DIMENSIONS:
            self.assertAlmostEqual(got[d], SIGMA, delta=1.0)

    def test_perturbation_stays_on_the_scale(self):
        import random
        from onet_scraper.uncertainty import DIMENSIONS, perturb
        rows = [{"dwa_id": "d1", **{d: v for d in DIMENSIONS}} for v in (0.0, 100.0)]
        out = perturb(rows, {d: 40.0 for d in DIMENSIONS}, random.Random(1))
        for r in out:
            for d in DIMENSIONS:
                self.assertGreaterEqual(r[d], 0.0)
                self.assertLessEqual(r[d], 100.0)

    def test_coerce_makes_csv_rows_arithmetic(self):
        """read_table hands back strings and propagate() averages without
        coercing, which raises inside statistics.fmean."""
        from onet_scraper.uncertainty import coerce
        out = coerce([{"dwa_id": "d1", "llm_exposure": "80", "importance": "55.5"}])
        self.assertEqual(out[0]["llm_exposure"], 80.0)
        self.assertEqual(out[0]["importance"], 55.5)

    def test_stability_flags_a_coin_flip(self):
        from onet_scraper.uncertainty import stability
        s = stability(["a"] * 4 + ["b"] * 6, "a")
        self.assertTrue(s["unstable"])
        self.assertEqual(s["modal"], "b")
        self.assertAlmostEqual(s["holds"], 0.4)

    def test_stability_accepts_a_solid_label(self):
        from onet_scraper.uncertainty import stability
        s = stability(["a"] * 39 + ["b"], "a")
        self.assertFalse(s["unstable"])
        self.assertAlmostEqual(s["holds"], 0.975)

    def test_summarise_brackets_the_draws(self):
        from onet_scraper.uncertainty import summarise
        s = summarise([float(i) for i in range(101)])
        self.assertAlmostEqual(s["mean"], 50.0, places=1)
        self.assertLessEqual(s["p05"], 10)
        self.assertGreaterEqual(s["p95"], 90)
        self.assertGreater(s["width"], 0)

    def test_the_real_report_is_internally_consistent(self):
        import json
        from pathlib import Path
        p = Path("data/out/uncertainty_report.json")
        if not p.exists():
            self.skipTest("bootstrap not run")
        r = json.loads(p.read_text())
        self.assertEqual(r["trials"], 400)
        self.assertLess(r["quadrant_unstable"], r["occupations"] * 0.1)
        for name, s in r["scenarios"].items():
            self.assertLessEqual(s["p05"], s["mean"])
            self.assertLessEqual(s["mean"], s["p95"])


# --------------------------------------------------------------------------- #
# the regression engine
#
# The reference values below were produced by statsmodels 0.15.0 / numpy 2.5.3
# on the fixture in this file (OLS with cov_type="HC1" and "cluster", WLS with
# cov_type="HC1", all with use_t=True) and are checked in as constants, because
# CI installs stdlib only and the engine has to be verifiable there. The full
# comparison - 36 checks including VIF, standardised coefficients and the
# fixed-effects equivalence on 400 simulated rows - is what these constants are
# a frozen sample of.
# --------------------------------------------------------------------------- #

from onet_scraper import regress  # noqa: E402

REG_Y = [12.0, 15.5, 9.25, 21.0, 18.75, 11.5, 24.0, 16.25, 13.0, 27.5, 19.25, 14.75]
REG_X1 = [1.0, 2.5, 0.5, 4.0, 3.5, 1.5, 5.0, 3.0, 2.0, 6.0, 4.5, 2.25]
REG_X2 = [200.0, 450.0, 125.0, 900.0, 725.0, 300.0, 1100.0, 600.0, 375.0,
          1400.0, 850.0, 425.0]
REG_X3 = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
REG_G = ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
REG_W = [1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 4.0, 1.0, 2.0, 3.0, 1.0, 2.0]
REG_TERMS = ["(intercept)", "x1", "x2", "x3"]

HC1_COEF = [8.2270259155, 0.0456757387, 0.0137484196, -0.0054907907]
HC1_SE = [0.697495765, 1.1744754217, 0.0049578459, 0.5706607951]
HC1_R2 = 0.9840273041
HC1_F = 261.10086418
CLUSTER_SE = [0.4363862649, 1.0229583424, 0.0045392323, 0.5180439459]
CLUSTER_F = 500.63047785
WLS_COEF = [8.0167536913, 0.5777862305, 0.0116598084, -0.0363255991]
WLS_SE = [0.6747132692, 1.2702406922, 0.0052354289, 0.5429279849]
WLS_R2 = 0.989090288


def _reg_design():
    return [[1.0, a, b, c] for a, b, c in zip(REG_X1, REG_X2, REG_X3)]


class TestRegressionEngine(unittest.TestCase):
    def test_hc1_matches_statsmodels(self):
        fit = regress.ols(REG_Y, _reg_design(), REG_TERMS)
        for got, want in zip(fit.coef, HC1_COEF):
            self.assertAlmostEqual(got, want, places=8)
        for got, want in zip(fit.se, HC1_SE):
            self.assertAlmostEqual(got, want, places=8)
        self.assertAlmostEqual(fit.r2, HC1_R2, places=9)
        self.assertAlmostEqual(fit.f_stat, HC1_F, places=5)
        self.assertEqual(fit.dof, 8)          # n - k, no clusters

    def test_cluster_matches_statsmodels(self):
        fit = regress.ols(REG_Y, _reg_design(), REG_TERMS, clusters=REG_G)
        for got, want in zip(fit.se, CLUSTER_SE):
            self.assertAlmostEqual(got, want, places=8)
        self.assertAlmostEqual(fit.f_stat, CLUSTER_F, places=5)
        self.assertEqual(fit.clusters, 4)
        self.assertEqual(fit.dof, 3)          # G - 1, not n - k

    def test_wls_matches_statsmodels(self):
        fit = regress.ols(REG_Y, _reg_design(), REG_TERMS, weights=REG_W)
        for got, want in zip(fit.coef, WLS_COEF):
            self.assertAlmostEqual(got, want, places=8)
        for got, want in zip(fit.se, WLS_SE):
            self.assertAlmostEqual(got, want, places=8)
        # The weighted R^2 has to be taken about the WEIGHTED mean. Centring on
        # the plain mean of the sqrt(w)-transformed values instead returns
        # ~0.999 for any employment-weighted fit, which is why this is pinned.
        self.assertAlmostEqual(fit.r2, WLS_R2, places=9)

    def test_integer_weights_equal_duplicated_rows(self):
        """Weighting a row by 3 has to be the same as listing it three times."""
        X = _reg_design()
        rep_y, rep_X = [], []
        for i, w in enumerate(REG_W):
            rep_y.extend([REG_Y[i]] * int(w))
            rep_X.extend([X[i]] * int(w))
        weighted = regress.ols(REG_Y, X, REG_TERMS, weights=REG_W)
        expanded = regress.ols(rep_y, rep_X, REG_TERMS)
        for got, want in zip(weighted.coef, expanded.coef):
            self.assertAlmostEqual(got, want, places=9)
        self.assertAlmostEqual(weighted.r2, expanded.r2, places=9)

    def test_bivariate_reduces_to_closed_form(self):
        fit = regress.ols(REG_Y, [[1.0, x] for x in REG_X1],
                          ["(intercept)", "x1"])
        mx = statistics.fmean(REG_X1)
        my = statistics.fmean(REG_Y)
        cov = sum((a - mx) * (b - my) for a, b in zip(REG_X1, REG_Y))
        var = sum((a - mx) ** 2 for a in REG_X1)
        self.assertAlmostEqual(fit.coef[1], cov / var, places=10)
        self.assertAlmostEqual(fit.coef[0], my - (cov / var) * mx, places=10)
        r = cov / (var * sum((b - my) ** 2 for b in REG_Y)) ** 0.5
        self.assertAlmostEqual(fit.r2, r * r, places=10)

    def test_within_transform_equals_dummy_variables(self):
        """Occupation fixed effects by demeaning == the same fit with dummies."""
        groups = sorted(set(REG_G))
        d = regress.Design(
            y=list(REG_Y),
            X=[[1.0, a, b] for a, b in zip(REG_X1, REG_X2)],
            terms=["(intercept)", "x1", "x2"],
        )
        demeaned = regress.within(d, REG_G)
        fe = regress.fit_design(demeaned, name="fe", outcome="y")
        dummies = [[a, b] + [1.0 if g == h else 0.0 for h in groups]
                   for a, b, g in zip(REG_X1, REG_X2, REG_G)]
        explicit = regress.ols(REG_Y, dummies,
                               ["x1", "x2"] + [f"g_{g}" for g in groups],
                               clusters=REG_G, has_intercept=False)
        for got, want in zip(fe.coef, explicit.coef[:2]):
            self.assertAlmostEqual(got, want, places=8)
        for got, want in zip(fe.se, explicit.se[:2]):
            self.assertAlmostEqual(got, want, places=8)
        self.assertEqual(fe.absorbed, len(groups))

    def test_singleton_clusters_collapse_to_hc1(self):
        """One observation per cluster: CR1's correction becomes HC1's exactly."""
        X = _reg_design()
        hc1 = regress.ols(REG_Y, X, REG_TERMS)
        singleton = regress.ols(REG_Y, X, REG_TERMS,
                                clusters=list(range(len(REG_Y))))
        for got, want in zip(singleton.se, hc1.se):
            self.assertAlmostEqual(got, want, places=12)
        self.assertNotEqual(singleton.dof, hc1.dof)   # but the dof still differ

    def test_constant_regressor_is_named_not_buried(self):
        X = [[1.0, x, 0.0] for x in REG_X1]
        with self.assertRaises(ValueError) as caught:
            regress.ols(REG_Y, X, ["(intercept)", "x1", "all_zero"])
        self.assertIn("all_zero", str(caught.exception))
        self.assertIn("constant", str(caught.exception))

    def test_vif_flags_collinearity(self):
        fit = regress.ols(REG_Y, _reg_design(), REG_TERMS)
        self.assertIsNone(fit.vif[0])                      # the intercept
        # x2 in the fixture is roughly a scaled x1, so its VIF is high on
        # purpose - about 49 - and a near-exact copy has to score far higher.
        self.assertGreater(fit.vif[1], 10)
        near_duplicate = [[1.0, x, x + 1e-3 * i, c]
                          for i, (x, c) in enumerate(zip(REG_X1, REG_X3))]
        collinear = regress.ols(REG_Y, near_duplicate,
                                ["(intercept)", "x1", "x1_copy", "x3"])
        self.assertGreater(collinear.vif[1], 100)

    def test_orthogonal_columns_have_unit_vif(self):
        X = [[1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [1.0, -1.0, 1.0],
             [1.0, -1.0, -1.0]]
        fit = regress.ols([1.0, 2.0, 3.0, 5.0], X, ["(intercept)", "a", "b"])
        self.assertAlmostEqual(fit.vif[1], 1.0, places=10)
        self.assertAlmostEqual(fit.vif[2], 1.0, places=10)

    def test_p_values_against_textbook_critical_values(self):
        self.assertAlmostEqual(regress.t_pvalue(2.228138852, 10), 0.05, places=8)
        self.assertAlmostEqual(regress.t_critical(10), 2.228138852, places=8)
        self.assertAlmostEqual(regress.f_pvalue(4.9646027, 1, 10), 0.05, places=7)
        # F(1, df) is t(df) squared, exactly.
        self.assertAlmostEqual(regress.f_pvalue(2.5 ** 2, 1, 17),
                               regress.t_pvalue(2.5, 17), places=12)
        self.assertAlmostEqual(regress.betainc(2.0, 3.0, 1.0), 1.0, places=12)
        self.assertEqual(regress.betainc(2.0, 3.0, 0.0), 0.0)

    def test_perfect_fit_and_zero_residual(self):
        X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
        fit = regress.ols([3.0, 5.0, 7.0], X, ["(intercept)", "x"])
        self.assertAlmostEqual(fit.coef[1], 2.0, places=10)
        self.assertAlmostEqual(fit.r2, 1.0, places=12)
        self.assertAlmostEqual(fit.rmse, 0.0, places=10)

    def test_refuses_more_parameters_than_rows(self):
        with self.assertRaises(ValueError):
            regress.ols([1.0, 2.0], [[1.0, 1.0], [1.0, 2.0]],
                        ["(intercept)", "x"])


class TestRegressionDesigns(unittest.TestCase):
    ROWS = [
        {"onet_soc_code": "15-0001.00", "susceptibility": 70.0,
         "importance": 80.0, "job_zone": 4},
        {"onet_soc_code": "15-0001.01", "susceptibility": 60.0,
         "importance": 50.0, "job_zone": 4},
        {"onet_soc_code": "29-0002.00", "susceptibility": 40.0,
         "importance": "", "job_zone": 3},          # dropped: no importance
        {"onet_soc_code": "29-0003.00", "susceptibility": 55.0,
         "importance": 60.0, "job_zone": None},     # dropped: no job zone
    ]

    def _design(self):
        return regress.design(
            self.ROWS,
            lambda r: regress._num(r.get("susceptibility")),
            {"importance": lambda r: regress._num(r.get("importance")),
             "job_zone": lambda r: regress._num(r.get("job_zone"))},
            cluster=lambda r: regress._soc6(r["onet_soc_code"]))

    def test_incomplete_rows_are_dropped_and_counted(self):
        d = self._design()
        self.assertEqual(len(d.y), 2)
        self.assertEqual(d.dropped, 2)
        self.assertEqual(d.terms, ["(intercept)", "importance", "job_zone"])

    def test_clusters_collapse_detail_occupations_to_one_soc(self):
        d = self._design()
        self.assertEqual(d.clusters, ["15-0001", "15-0001"])

    def test_soc6_strips_the_detail_suffix(self):
        self.assertEqual(regress._soc6("15-2051.01"), "15-2051")
        self.assertEqual(regress._soc6("15-2051"), "15-2051")

    def test_within_subtracts_group_means_and_drops_intercept(self):
        d = regress.Design(y=[10.0, 20.0, 40.0, 60.0],
                           X=[[1.0, 1.0], [1.0, 3.0], [1.0, 5.0], [1.0, 9.0]],
                           terms=["(intercept)", "x"])
        out = regress.within(d, ["a", "a", "b", "b"])
        self.assertEqual(out.terms, ["x"])
        self.assertEqual(out.y, [-5.0, 5.0, -10.0, 10.0])
        self.assertEqual([row[0] for row in out.X], [-1.0, 1.0, -2.0, 2.0])
        self.assertEqual(out.absorbed, 2)
        self.assertFalse(out.has_intercept)

    def test_index_reliability_is_signal_over_total_variance(self):
        rows = [{"susceptibility_mean": m, "susceptibility_sd": 1.0}
                for m in range(10, 110, 5)]
        out = regress.index_reliability(rows)
        expected = 1.0 - 1.0 / statistics.variance([r["susceptibility_mean"]
                                                    for r in rows])
        self.assertAlmostEqual(out["reliability"], round(expected, 4), places=4)
        self.assertIn("calibration", out["what_it_does_not_cover"])

    def test_index_reliability_needs_a_sample(self):
        self.assertIsNone(regress.index_reliability([]))

    def test_build_reports_what_it_could_not_run(self):
        report = regress.build({"occupation_susceptibility": [], "fits": []})
        self.assertEqual(report["fits"], [])
        self.assertEqual(len(report["skipped"]), 5)
        self.assertTrue(any("fetch-descriptors" in s for s in report["skipped"]))

    def test_nested_reports_both_directions(self):
        X = [[1.0, a, b] for a, b in zip(REG_X1, REG_X2)]
        out = regress.nested(REG_Y, X, ["(intercept)", "x1", "x2"], ["x2"])
        self.assertEqual(out["dropped"], ["x2"])
        self.assertGreater(out["r2_full"], out["r2_restricted"])
        self.assertAlmostEqual(out["r2_gain"],
                               out["r2_full"] - out["r2_restricted"], places=4)
