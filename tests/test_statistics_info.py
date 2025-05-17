import pytest
import numpy as np
import pandas as pd

from statshelper import StatisticsInfo
from statshelper.tools.validators import validate_and_convert_array

class TestStatisticsInfo:
    # Test Utilities
    def test_validate_and_convert_array_valid(self):
        # Test with list
        result = validate_and_convert_array([1, 2, 3])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))
        
        # Test with pandas Series
        result = validate_and_convert_array(pd.Series([4, 5, 6]))
        np.testing.assert_array_equal(result, np.array([4, 5, 6]))
        
        # Test with numpy array - now expecting float conversion
        input_arr = np.array([7, 8, 9])
        result = validate_and_convert_array(input_arr)
        np.testing.assert_array_equal(result, np.array([7., 8., 9.]))
    
    def test_validate_and_convert_array_invalid(self):
        # Test empty input
        with pytest.raises(ValueError):
            validate_and_convert_array([])
            
        # Test non-numeric input
        with pytest.raises(ValueError):
            validate_and_convert_array(['a', 'b', 'c'])
            
        # Test mixed numeric and non-numeric
        with pytest.raises(ValueError):
            validate_and_convert_array([1, 2, 'three'])
    
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