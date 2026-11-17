import unittest

import pandas as pd

from app import _benford_analysis, _chi_square_result, _ks_two_sample, _leading_digits


class AnalysisTests(unittest.TestCase):
    def test_leading_digits_handles_small_values_and_trailing_integer_zeroes(self):
        self.assertEqual(_leading_digits(0.00456, 2), 45)
        self.assertEqual(_leading_digits(1200, 2), 12)
        self.assertIsNone(_leading_digits(0, 1))
        self.assertIsNone(_leading_digits(5, 2))

    def test_benford_analysis_excludes_nonpositive_and_counts_two_digits(self):
        result = _benford_analysis(pd.Series([12, 19, 0, -30, None]), 2)
        self.assertEqual(result["Observed count"].sum(), 2)
        self.assertEqual(result.loc[result["Digits"] == 12, "Observed count"].iloc[0], 1)
        self.assertAlmostEqual(result["Benford expected"].sum(), 1.0)

    def test_chi_square_reports_when_sample_is_too_small(self):
        result = _chi_square_result(_benford_analysis(pd.Series([1, 2]), 1))
        self.assertIsNone(result["p-value"])

    def test_ks_two_sample_handles_different_sample_lengths(self):
        result = _ks_two_sample(pd.Series([1, 2, 3]), pd.Series([10, 11]))
        self.assertEqual(result["Statistic"], 1.0)
        self.assertEqual(result["Sample 1 size"], 3)
        self.assertEqual(result["Sample 2 size"], 2)
        self.assertGreaterEqual(result["p-value (approx.)"], 0.0)
        self.assertLessEqual(result["p-value (approx.)"], 1.0)

    def test_ks_reports_significant_difference_for_separated_samples(self):
        result = _ks_two_sample(pd.Series(range(20)), pd.Series(range(30, 50)))
        self.assertEqual(result["Statistic"], 1.0)
        self.assertLess(result["p-value (approx.)"], 0.05)


if __name__ == "__main__":
    unittest.main()
