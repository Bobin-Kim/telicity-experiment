import json
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class DeepSeekRQ25RQ26Contract(unittest.TestCase):
    def test_rq25_completed_reaudit(self):
        out = ROOT / "results/deepseek-7b-telicity-rq/telicity_decision_control"
        if not (out / "audit.json").exists():
            self.skipTest("completed DeepSeek RQ2.5 reference outputs are not installed")
        self.assertEqual(json.loads((out / "audit.json").read_text())["status"], "PASS")
        validation = json.loads((out / "handoff_validation.json").read_text())
        self.assertEqual(validation["status"], "PASS")
        self.assertEqual(validation["row_counts"]["raw"], 17200)

    def test_rq26_plan_is_cache_derived_and_continuous(self):
        out = ROOT / "results/deepseek-7b-rq2/rq2_6"
        if not (out / "execution_plan.json").exists():
            self.skipTest("completed DeepSeek RQ2.6 reference outputs are not installed")
        plan = json.loads((out / "execution_plan.json").read_text())
        self.assertEqual(plan["status"], "PASS")
        self.assertEqual(plan["scoring"], "m = z(True_first_token) - z(Unknown_first_token)")
        self.assertEqual(plan["candidate_layers"], list(range(plan["generator_layer"])))
        self.assertEqual(plan["reused"]["rq25_audit"], "PASS")
        axes = np.load(out / "fold_axes.npz", allow_pickle=False)
        self.assertEqual(len(axes.files), 30)
        self.assertTrue(all(np.isfinite(axes[key]).all() for key in axes.files))

    def test_rq26_source_contains_required_controls(self):
        source = (ROOT / "src/rq2/deepseek_rq26.py").read_text().replace(" ", "")
        for condition in ("C_to_D", "D_to_C", "random_C_to_D", "random_D_to_C"):
            self.assertIn(condition, source)
        self.assertIn('"binary_artifacts_used":False', source)
        self.assertIn('logits[local,0]-logits[local,2]', source)
        self.assertIn("restore_generator", source)

    def test_rq26_completed_audit_and_interpretation_boundary(self):
        out = ROOT / "results/deepseek-7b-rq2/rq2_6"
        if not (out / "audit.json").exists():
            self.skipTest("completed DeepSeek RQ2.6 reference outputs are not installed")
        audit = json.loads((out / "audit.json").read_text())
        self.assertEqual(audit["status"], "PASS")
        self.assertFalse(audit["binary_artifacts_used"])
        self.assertTrue(all(audit["checks"].values()))
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["status"], "complete")
        self.assertIn("GENERATOR_ROUTED", summary["classification"])
        self.assertFalse(summary["fraction_mediated_interpretable"])
        self.assertTrue(summary["all_bidirectional_margin_actuators_outside_validated_generator_route"])


if __name__ == "__main__":
    unittest.main()
