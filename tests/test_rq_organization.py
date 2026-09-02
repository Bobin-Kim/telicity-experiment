import json
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import load_rq_config, registered_aliases
from src.common.contracts import CANONICAL_FORMULA, RQ2_SUBQUESTIONS
from src.rq1.routing import rq1_stages
from src.rq2.routing import resolve_subquestion


class ResearchQuestionOrganization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rq1, cls.base1 = load_rq_config(ROOT / "configs/rq1_telicity_reemergence.yaml", "rq1")
        cls.rq2, cls.base2 = load_rq_config(ROOT / "configs/rq2_decision_mechanism.yaml", "rq2")

    def test_shared_registry_and_score(self):
        self.assertEqual(self.base1, self.base2)
        aliases = registered_aliases(self.base1)
        self.assertIn("deepseek-7b-chat", aliases)
        self.assertIn("mistral-7b-v03", aliases)
        self.assertIn("qwen2.5-7b", aliases)
        self.assertEqual(self.rq1["scoring_contract"]["formula"], CANONICAL_FORMULA)
        self.assertEqual(self.rq2["scoring_contract"]["formula"], CANONICAL_FORMULA)

    def test_rq1_route_is_isolated(self):
        locked = rq1_stages(False)
        extended = rq1_stages(True)
        self.assertNotIn("source_pathway_extended", locked)
        self.assertIn("source_pathway_extended", extended)
        for stages in (locked, extended):
            self.assertFalse(set(stages) & {"behavior", "predictive_localization", "causal_intervention",
                                            "component_analysis", "generator_localization", "aggregate",
                                            "upstream_source"})
        self.assertFalse(self.rq1["execution"]["automatic_rq2"])

    def test_all_six_rq2_routes_are_explicit(self):
        self.assertEqual(set(self.rq2["subquestions"]), set(RQ2_SUBQUESTIONS))
        with self.assertRaises(ValueError):
            resolve_subquestion(self.rq2, "")
        for selector, identifier in RQ2_SUBQUESTIONS.items():
            route = resolve_subquestion(self.rq2, selector)
            self.assertEqual(route["id"], identifier)
            self.assertNotIn("upstream_source", route["generic_stages"])
            self.assertFalse(set(route["generic_stages"]) & {"reemergence_cache", "reemergence_sweep",
                                                               "source_pathway", "source_pathway_extended",
                                                               "reemergence_aggregate"})

    def test_executable_configs_have_no_model_findings(self):
        for path in (ROOT / "configs/rq1_telicity_reemergence.yaml",
                     ROOT / "configs/rq2_decision_mechanism.yaml"):
            value = yaml.safe_load(path.read_text())
            text = json.dumps({"execution": value.get("execution"),
                               "subquestions": value.get("subquestions"),
                               "logical_stages": value.get("logical_stages")}).lower()
            self.assertNotIn("binary", text)
            for prohibited in ("effect_size", "p_value", "successful_layer", "conclusion"):
                self.assertNotIn(prohibited, text)

    def test_manifests_and_invalid_mistral_boundary(self):
        baseline = json.loads((ROOT / "manifests/result_hash_baseline.json").read_text())
        all_manifest = json.loads((ROOT / "manifests/all_artifacts.json").read_text())
        rq1 = json.loads((ROOT / "manifests/rq1_artifacts.json").read_text())
        rq2 = json.loads((ROOT / "manifests/rq2_artifacts.json").read_text())
        self.assertEqual(baseline["file_count"], len(baseline["files"]))
        local_paths = {row["existing_path"] for row in all_manifest["artifacts"]
                       if row["artifact_kind"] == "file"}
        self.assertEqual(len(local_paths), baseline["file_count"])
        required = {"model_alias", "research_question", "subquestion", "stage", "protocol_type",
                    "execution_status", "audit_status", "existing_path", "sha256", "classification"}
        self.assertTrue(all(required <= set(row) for row in all_manifest["artifacts"]))
        self.assertTrue(any(row["stage"] == "source_pathway_locked" and row["classification"] == "RQ1 official"
                            for row in rq1["artifacts"]))
        self.assertTrue(any(row["stage"] == "source_pathway_extension"
                            and row["classification"] == "RQ1 extension/unexecuted"
                            for row in rq1["artifacts"]))
        failed = [row for row in rq2["artifacts"] if "mistral7b_decision_alignment_A36" in row["existing_path"]
                  or "mistral7b_upstream_true_origin_A37" in row["existing_path"]]
        self.assertEqual(len(failed), 2)
        self.assertTrue(all(row["classification"] == "invalid" and row["audit_status"] == "FAIL"
                            for row in failed))
        self.assertTrue(any(row["model_alias"] == "mistral-7b-v03"
                            and row["subquestion"] == "rq2.true_driving_origin"
                            and row["classification"] == "RQ2 planned/unexecuted"
                            and row["execution_status"] == "planned" for row in rq2["artifacts"]))
        self.assertTrue(any(row["model_alias"] == "deepseek-7b-chat"
                            and row["subquestion"] == "rq2.telicity_decision_relationship"
                            and row["classification"] == "RQ2.5 official" for row in rq2["artifacts"]))


if __name__ == "__main__":
    unittest.main()
