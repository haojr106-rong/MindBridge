import unittest

from app.risk_eval.runner import classification_metrics


class RiskEvaluationMetricsTest(unittest.TestCase):
    def test_three_class_metrics_and_high_risk_rates(self):
        actual = ["LOW", "LOW", "MEDIUM", "MEDIUM", "HIGH", "HIGH"]
        predicted = ["LOW", "HIGH", "MEDIUM", "LOW", "HIGH", "MEDIUM"]

        metrics = classification_metrics(actual, predicted)

        self.assertEqual(metrics["highRiskRecall"], 0.5)
        self.assertEqual(metrics["highRiskMissRate"], 0.5)
        self.assertEqual(metrics["highRiskFalsePositiveRate"], 0.25)
        self.assertEqual(metrics["highRiskFalseDiscoveryRate"], 0.5)
        self.assertEqual(
            metrics["confusionMatrix"]["rowsActualColumnsPredicted"]["HIGH"],
            {"LOW": 0, "MEDIUM": 1, "HIGH": 1},
        )
        self.assertEqual(metrics["perClass"]["MEDIUM"]["precision"], 0.5)
        self.assertEqual(metrics["perClass"]["MEDIUM"]["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
