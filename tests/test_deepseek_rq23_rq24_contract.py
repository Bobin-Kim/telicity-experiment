import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RQ23 = ROOT / "results/deepseek-7b-rq2/rq2_3"
RQ24 = ROOT / "results/deepseek-7b-rq2/rq2_4"


class DeepSeekRQ23RQ24Contract(unittest.TestCase):
    def test_backend_keeps_continuous_first_token_contract(self):
        source = (ROOT / "src/rq2/deepseek_rq23_rq24.py").read_text()
        compact = source.replace(" ", "")
        self.assertIn('LABELS=("True","False","Unknown")', compact)
        self.assertIn('logits[local,0]-logits[local,2]', compact)
        self.assertIn('"binary_artifacts_used":False', compact)

    def test_rq23_completed_audit_and_controls(self):
        if not (RQ23 / "audit.json").exists():
            self.skipTest("completed DeepSeek RQ2.3 reference outputs are not installed")
        audit = json.loads((RQ23 / "audit.json").read_text())
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["rows"], 6000)
        self.assertFalse(audit["binary_artifacts_used"])
        self.assertTrue(audit["checks"]["five_folds"])
        self.assertTrue(audit["checks"]["matched_same_state_control"])
        self.assertTrue(audit["checks"]["restore_controls"])
        summary = json.loads((RQ23 / "summary.json").read_text())
        self.assertEqual(summary["status"], "complete")
        self.assertTrue(summary["generator"]["reliable_causal"])
        self.assertGreater(summary["generator"]["ci95_low"], 0)

    def test_rq24_completed_heldout_audit(self):
        if not (RQ24 / "audit.json").exists():
            self.skipTest("completed DeepSeek RQ2.4 reference outputs are not installed")
        audit = json.loads((RQ24 / "audit.json").read_text())
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["rows"], 3600)
        self.assertEqual(audit["True_token_id"], 8173)
        self.assertEqual(audit["Unknown_token_id"], 21483)
        self.assertFalse(audit["binary_artifacts_used"])
        self.assertTrue(audit["checks"]["heldout_selection"])
        self.assertTrue(audit["checks"]["exact_restore"])
        with (RQ24 / "selected_hyperparameters.csv").open() as handle:
            selected = list(csv.DictReader(handle))
        self.assertEqual(len(selected), 15)
        self.assertEqual({int(row["fold"]) for row in selected}, set(range(5)))
        self.assertEqual({row["variant"] for row in selected}, {"hard", "soft", "product"})

    def test_manifest_routes_completed_artifacts(self):
        manifest = json.loads((ROOT / "manifests/rq2_artifacts.json").read_text())
        official = {(row["model_alias"], row["subquestion"], row["classification"])
                    for row in manifest["artifacts"]
                    if row["execution_status"] == "complete" and row["audit_status"] == "PASS"}
        self.assertIn(("deepseek-7b-chat", "rq2.decision_circuit", "RQ2.3 official"), official)
        self.assertIn(("deepseek-7b-chat", "rq2.selective_group_c_correction", "RQ2.4 official"), official)


if __name__ == "__main__":
    unittest.main()
