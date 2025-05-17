import pytest
import numpy as np

from statshelper import Hypothesis

class TestHypothesis:
    def test_calculate_mean_pvalue(self):
        # Z-test two-sided
        p_z = Hypothesis.calculate_mean_pvalue(
            data_size=100,
            sample_mean=105,
            std_dev=15,
            significance=0.05,
            hypothesis_mean=100,
            alternative="two-sided",
            is_sample_std=False
        )
        assert p_z == pytest.approx(0.00085812, abs=0.0001)
        
        # t-test right-tailed
        p_t = Hypothesis.calculate_mean_pvalue(
            data_size=20,
            sample_mean=22,
            std_dev=5,
            significance=0.05,
            hypothesis_mean=20,
            alternative="right",
            is_sample_std=True
        )
        assert p_t == pytest.approx(0.0447, abs=0.001)

    def test_calculate_proportion_pvalue(self):
        # Left-tailed test
        p_left = Hypothesis.calculate_proportion_pvalue(
            data_size=1000,
            sample_proportion=0.48,
            significance=0.05,
            hypothesis_proportion=0.5,
            alternative="left"
        )
        assert p_left == pytest.approx(0.102, abs=0.001)
        
        # Two-sided test
        p_two = Hypothesis.calculate_proportion_pvalue(
            data_size=1000,
            sample_proportion=0.53,
            significance=0.05,
            hypothesis_proportion=0.5,
            alternative="two-sided"
        )
        assert p_two == pytest.approx(0.05777, abs=0.001)

    def test_compare_variances_f_test(self):
        """Test with equal variances (should not reject null)."""
        p_value = Hypothesis.compare_variances_f_test(
            variance_1=4.0,
            variance_2=4.0,
            size_1=30,
            size_2=30,
            significance=0.05,
            alternative="two-sided"
        )
        assert p_value == 1.0  # Exactly equal variances should give p=1

        """Test with clearly unequal variances (should reject null)."""
        p_value = Hypothesis.compare_variances_f_test(
            variance_1=9.0,
            variance_2=1.0,
            size_1=30,
            size_2=30,
            significance=0.05,
            alternative="two-sided"
        )
        assert p_value < 0.05

        """Test right-tailed alternative (v1 > v2)."""
        p_value = Hypothesis.compare_variances_f_test(
            variance_1=5.0,
            variance_2=2.0,
            size_1=20,
            size_2=20,
            significance=0.05,
            alternative="right"
        )
        # Should be significant since 5 > 2
        assert p_value < 0.05

        """Test left-tailed alternative (v1 < v2)."""
        p_value = Hypothesis.compare_variances_f_test(
            variance_1=2.0,
            variance_2=5.0,
            size_1=20,
            size_2=20,
            significance=0.05,
            alternative="left"
        )
        # Should be significant since 2 < 5
        assert p_value < 0.05

    def test_compare_means_t_test(self):
        data_1 = [13, 19, 14, 17, 21, 24, 10, 14, 13, 15]
        data_2 = [16, 14, 19, 18, 19, 20, 15, 18, 17, 18]
        p_value = Hypothesis.compare_means_t_test(
            data_1=data_1,
            data_2=data_2,
            significance=0.05,
            alternative='two-sided',
            paired=False
        )

        assert p_value == pytest.approx(0.35892, abs=0.01)

    def test_compare_proportions_z_test(self):
        p_value = Hypothesis.compare_proportions_z_test(
            success_count_1=26,
            success_count_2=54,
            size_1=40,
            size_2=60,
            significance=0.05,
            alternative="two-sided",
        )
        assert p_value == pytest.approx(0.00220, abs=0.001)

    def test_normal_distribution(self):
        """Test with normally distributed data."""
        # Generate normal data
        np.random.seed(42)
        normal_data = np.random.normal(loc=0, scale=1, size=100)
        
        results = Hypothesis.normality_tests(normal_data)
        
        # Check all tests are present
        assert set(results.keys()) == {'shapiro-wilk', 'kolmogorov-smirnov', 'anderson-darling'}
        
        # Shapiro-Wilk - should not reject normality
        assert 0.9 <= results['shapiro-wilk']['statistic'] <= 1.0
        assert results['shapiro-wilk']['p_value'] > 0.05
        
        # Kolmogorov-Smirnov - should not reject normality
        assert results['kolmogorov-smirnov']['statistic'] < 0.1
        assert results['kolmogorov-smirnov']['p_value'] > 0.05
        
        # Anderson-Darling - test statistic should be less than critical value at 5%
        critical_5pct = results['anderson-darling']['critical_values'][2]  # 5% is at index 2
        assert results['anderson-darling']['statistic'] < critical_5pct

    def test_non_normal_distribution(self):
        """Test with clearly non-normal data (exponential)."""
        # Generate exponential data
        np.random.seed(42)
        exp_data = np.random.exponential(scale=1.0, size=100)
        
        results = Hypothesis.normality_tests(exp_data)
        
        # Shapiro-Wilk - should reject normality
        assert results['shapiro-wilk']['statistic'] < 0.9
        assert results['shapiro-wilk']['p_value'] <= 0.05
        
        # Kolmogorov-Smirnov - should reject normality
        assert results['kolmogorov-smirnov']['statistic'] > 0.1
        assert results['kolmogorov-smirnov']['p_value'] <= 0.05
        
        # Anderson-Darling - test statistic should exceed critical value at 5%
        critical_5pct = results['anderson-darling']['critical_values'][2]
        assert results['anderson-darling']['statistic'] > critical_5pct
