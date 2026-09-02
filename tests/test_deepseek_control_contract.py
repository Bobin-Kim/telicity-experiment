import csv
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation.run_deepseek_telicity_decision_control import (
    CONDITIONS_PER_EXAMPLE,
    DECISION_LAYER,
    DOSES,
    EXPECTED_INTERVENTION_ROWS,
    EXPECTED_ITEMS,
    TELICITY_LAYER,
    conditions_for_coordinate,
)


class DeepSeekControlContract(unittest.TestCase):
    def test_locked_design(self):
        self.assertEqual(DOSES, (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0))
        self.assertEqual((TELICITY_LAYER, DECISION_LAYER), (19, 30))
        self.assertEqual(CONDITIONS_PER_EXAMPLE, 43)
        self.assertEqual(EXPECTED_INTERVENTION_ROWS, 17_200)
        self.assertEqual(1 + sum(len(conditions_for_coordinate(x))
                                 for x in ("L19", "L30", "final_norm")), 43)

    def test_control_grid(self):
        l19 = conditions_for_coordinate("L19")
        l30 = conditions_for_coordinate("L30")
        final = conditions_for_coordinate("final_norm")
        self.assertEqual(len(l19), 14)
        self.assertEqual(len(l30), 14)
        self.assertEqual(len(final), 14)
        self.assertEqual(sum(x[1] == "random" for x in l19), 6)
        self.assertEqual(sum(x[1] == "label_permutation" for x in l19), 2)
        self.assertEqual(sum(x[1] == "random" for x in l30), 6)
        self.assertEqual(sum(x[1] == "label_permutation" for x in l30), 2)
        self.assertEqual(sum(x[0] == "telicity_final_norm" and x[1] == "off_target" for x in final), 2)

    def test_completed_reference_if_present(self):
        out = ROOT / "results/deepseek-7b-telicity-rq/telicity_decision_control"
        validation = out / "handoff_validation.json"
        if not validation.exists():
            self.skipTest("completed DeepSeek reference outputs are not installed")
        result = json.loads(validation.read_text())
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["scoring_rule"], "m = z(True_first_token) - z(Unknown_first_token)")
        self.assertEqual((result["True_token_id"], result["Unknown_token_id"]), (8173, 21483))
        with (out / "interventions_all_examples.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), EXPECTED_INTERVENTION_ROWS)
        self.assertEqual(len({(r["fold"], r["example_id"], r["coordinate"], r["representation"],
                               r["control"], r["dose"]) for r in rows}), EXPECTED_INTERVENTION_ROWS)
        self.assertEqual(len({r["example_id"] for r in rows if r["representation"] == "baseline"}),
                         EXPECTED_ITEMS)


if __name__ == "__main__":
    unittest.main()
