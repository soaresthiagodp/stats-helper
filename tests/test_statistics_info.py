import pytest
import numpy as np
import pandas as pd

from statshelper import StatisticsInfo

class TestStatisticsInfo:
    # Test Utilities
    def test_validate_and_convert_array_valid(self):
        # Test with list
        result = StatisticsInfo._validate_and_convert_array([1, 2, 3])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))
        
        # Test with pandas Series
        result = StatisticsInfo._validate_and_convert_array(pd.Series([4, 5, 6]))
        np.testing.assert_array_equal(result, np.array([4, 5, 6]))
        
        # Test with numpy array - now expecting float conversion
        input_arr = np.array([7, 8, 9])
        result = StatisticsInfo._validate_and_convert_array(input_arr)
        np.testing.assert_array_equal(result, np.array([7., 8., 9.]))
    
    def test_validate_and_convert_array_invalid(self):
        # Test empty input
        with pytest.raises(ValueError):
            StatisticsInfo._validate_and_convert_array([])
            
        # Test non-numeric input
        with pytest.raises(ValueError):
            StatisticsInfo._validate_and_convert_array(['a', 'b', 'c'])
            
        # Test mixed numeric and non-numeric
        with pytest.raises(ValueError):
            StatisticsInfo._validate_and_convert_array([1, 2, 'three'])
    
    def test_get_statistics(self):
        data = [10, 12, 14, 16, 18]
        result = StatisticsInfo.get_statistics(data)
        
        assert result['mean'] == 14.0
        assert result['std'] == pytest.approx(2.8284271247461903)
        assert result['count'] == 5
        assert result['min'] == 10
        assert result['max'] == 18
        assert result['q1'] == 12
        assert result['q2'] == 14
        assert result['q3'] == 16
        assert result['iqr'] == 4
        
        # Test sample vs population variance
        sample_result = StatisticsInfo.get_statistics(data, is_sample=True)
        pop_result = StatisticsInfo.get_statistics(data, is_sample=False)
        assert sample_result['variance'] == 10.0
        assert pop_result['variance'] == 8.0
        
        # Test z-score return
        result_with_z = StatisticsInfo.get_statistics(data, return_z_score=True)
        assert 'z_score' in result_with_z
        assert len(result_with_z['z_score']) == 5
    
    def test_get_z_critical(self):
        assert StatisticsInfo.get_z_critical(0.95) == pytest.approx(1.96, abs=0.01)
        assert StatisticsInfo.get_z_critical(0.99) == pytest.approx(2.576, abs=0.01)
        
    def test_get_t_critical(self):
        # Compare with known t-values
        assert StatisticsInfo.get_t_critical(0.95, df=10) == pytest.approx(2.228, abs=0.01)
        assert StatisticsInfo.get_t_critical(0.99, df=20) == pytest.approx(2.845, abs=0.01)
        
    def test_calculate_bin_width(self):
        data = np.arange(100)
        num_bins, bin_width = StatisticsInfo.calculate_bin_width(data)
        
        # Verify Sturges Rule calculation
        assert num_bins == 8
        assert bin_width == pytest.approx(12.38, abs=0.01)
        
        # Test edge case with small dataset
        small_data = [1, 2, 3]
        small_bins, small_width = StatisticsInfo.calculate_bin_width(small_data)
        assert small_bins == 3
        assert small_width == pytest.approx(0.67, abs=0.01)
    
    # Test Mean Related Functions
    def test_calculate_mean_sample_size(self):
        # Infinite population
        n = StatisticsInfo.calculate_mean_sample_size(0.95, 2, 5)
        assert n == 25  # (1.96*5/2)^2 = 24.01 -> ceil to 25
        
        # Finite population
        n_finite = StatisticsInfo.calculate_mean_sample_size(0.95, 2, 5, population_size=100)
        assert n_finite == 20
        
    def test_calculate_mean_error(self):
        # Z-test
        error_z = StatisticsInfo.calculate_mean_error(0.95, 30, 5, is_sample_std=False)
        assert error_z == pytest.approx(1.789, abs=0.01)
        
        # t-test
        error_t = StatisticsInfo.calculate_mean_error(0.95, 30, 5, is_sample_std=True)
        assert error_t == pytest.approx(1.871, abs=0.01)
        
        # Test invalid inputs
        with pytest.raises(ValueError):
            StatisticsInfo.calculate_mean_error(1.5, 30, 5)  # invalid confidence
            
    def test_calculate_mean_pvalue(self):
        # Z-test two-sided
        p_z = StatisticsInfo.calculate_mean_pvalue(
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
        p_t = StatisticsInfo.calculate_mean_pvalue(
            data_size=20,
            sample_mean=22,
            std_dev=5,
            significance=0.05,
            hypothesis_mean=20,
            alternative="right",
            is_sample_std=True
        )
        assert p_t == pytest.approx(0.0447, abs=0.001)
    
    # Test Proportion Related Functions
    def test_calculate_proportion_sample_size(self):
        # Infinite population
        n = StatisticsInfo.calculate_proportion_sample_size(0.95, 0.05, 0.5)
        assert n == 385  # (1.96^2 * 0.5 * 0.5) / 0.05^2 = 384.16 -> 385
        
        # Finite population
        n_finite = StatisticsInfo.calculate_proportion_sample_size(0.95, 0.05, 0.5, population_size=1000)
        assert n_finite == 278
        
    def test_calculate_proportion_error(self):
        error = StatisticsInfo.calculate_proportion_error(0.95, 1000, 0.5)
        assert error == pytest.approx(0.031, abs=0.001)
        
        # Test invalid proportion
        with pytest.raises(ValueError):
            StatisticsInfo.calculate_proportion_error(0.95, 1000, 1.5)
            
        # Test small np
        with pytest.raises(ValueError):
            StatisticsInfo.calculate_proportion_error(0.95, 10, 0.1)
    
    def test_calculate_proportion_pvalue(self):
        # Left-tailed test
        p_left = StatisticsInfo.calculate_proportion_pvalue(
            data_size=1000,
            sample_proportion=0.48,
            significance=0.05,
            hypothesis_proportion=0.5,
            alternative="left"
        )
        assert p_left == pytest.approx(0.102, abs=0.001)
        
        # Two-sided test
        p_two = StatisticsInfo.calculate_proportion_pvalue(
            data_size=1000,
            sample_proportion=0.53,
            significance=0.05,
            hypothesis_proportion=0.5,
            alternative="two-sided"
        )
        assert p_two == pytest.approx(0.05777, abs=0.001)

    def test_compare_variances_f_test(self):
        """Test with equal variances (should not reject null)."""
        p_value = StatisticsInfo.compare_variances_f_test(
            variance_1=4.0,
            variance_2=4.0,
            size_1=30,
            size_2=30,
            significance=0.05,
            alternative="two-sided"
        )
        assert p_value == 1.0  # Exactly equal variances should give p=1

        """Test with clearly unequal variances (should reject null)."""
        p_value = StatisticsInfo.compare_variances_f_test(
            variance_1=9.0,
            variance_2=1.0,
            size_1=30,
            size_2=30,
            significance=0.05,
            alternative="two-sided"
        )
        assert p_value < 0.05

        """Test right-tailed alternative (v1 > v2)."""
        p_value = StatisticsInfo.compare_variances_f_test(
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
        p_value = StatisticsInfo.compare_variances_f_test(
            variance_1=2.0,
            variance_2=5.0,
            size_1=20,
            size_2=20,
            significance=0.05,
            alternative="left"
        )
        # Should be significant since 2 < 5
        assert p_value < 0.05

    def test_normal_distribution(self):
        """Test with normally distributed data."""
        # Generate normal data
        np.random.seed(42)
        normal_data = np.random.normal(loc=0, scale=1, size=100)
        
        results = StatisticsInfo.normality_tests(normal_data)
        
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
        
        results = StatisticsInfo.normality_tests(exp_data)
        
        # Shapiro-Wilk - should reject normality
        assert results['shapiro-wilk']['statistic'] < 0.9
        assert results['shapiro-wilk']['p_value'] <= 0.05
        
        # Kolmogorov-Smirnov - should reject normality
        assert results['kolmogorov-smirnov']['statistic'] > 0.1
        assert results['kolmogorov-smirnov']['p_value'] <= 0.05
        
        # Anderson-Darling - test statistic should exceed critical value at 5%
        critical_5pct = results['anderson-darling']['critical_values'][2]
        assert results['anderson-darling']['statistic'] > critical_5pct

    # Test Outlier Detection
    def test_get_outlier_limit(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])
        low, high = StatisticsInfo.get_outlier_limit(s)
        assert low == -4.0
        assert high == 16.0
        
    def test_detect_outliers(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])
        
        # IQR method
        outliers_iqr = StatisticsInfo.detect_outliers(s, method='iqr')
        assert outliers_iqr.sum() == 1
        assert outliers_iqr.iloc[-1]
        
        # Z-score method
        outliers_z = StatisticsInfo.detect_outliers(s, method='zscore')
        assert outliers_z.sum() == 1
        
        # MAD method
        outliers_mad = StatisticsInfo.detect_outliers(s, method='mad')
        assert outliers_mad.sum() == 1
        
        # Test invalid method
        with pytest.raises(ValueError):
            StatisticsInfo.detect_outliers(s, method='invalid')