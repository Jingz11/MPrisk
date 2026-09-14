import unittest
from pathlib import Path

import pandas as pd

from src.common import load_config
from src.ech import compute_scores_by_system, scenario_summary


ROOT = Path(__file__).resolve().parents[1]


class CoreWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observations = pd.read_csv(ROOT / "data" / "demo_observations.csv")
        cls.config = load_config(ROOT / "config" / "ech_parameters.json")
        cls.scores = compute_scores_by_system(cls.observations, cls.config)

    def test_demo_keys_and_counts(self):
        self.assertEqual(len(self.observations), 160)
        self.assertEqual(self.observations["No.ID"].nunique(), 160)
        self.assertTrue(self.observations["No.ID"].str.startswith("DEMO-").all())
        self.assertEqual(self.observations["Class"].value_counts().to_dict(), {"river": 80, "lake": 80})

    def test_ech_names_and_range(self):
        self.assertIn("H", self.scores)
        self.assertNotIn("T", self.scores)
        ranges = self.scores.groupby("Class")["MP_Risk"].agg(["min", "max"])
        self.assertTrue((ranges["min"].abs() < 1e-10).all())
        self.assertTrue(((ranges["max"] - 100).abs() < 1e-10).all())

    def test_scenario_direction(self):
        table = scenario_summary(self.scores).set_index(["System", "Scenario"])
        for system in ["river", "lake"]:
            baseline = table.loc[(system, "Baseline"), "Mean_MP_Risk"]
            self.assertGreater(table.loc[(system, "High Exposure"), "Mean_MP_Risk"], baseline)
            self.assertGreater(table.loc[(system, "Increased Interactions"), "Mean_MP_Risk"], baseline)
            self.assertLess(table.loc[(system, "Best Case"), "Mean_MP_Risk"], baseline)


if __name__ == "__main__":
    unittest.main()
