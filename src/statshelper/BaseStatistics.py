import math
from functools import partial
from typing import Literal

import pandas as pd
import numpy as np
from scipy.stats import norm, t, zscore

class StatisticsInfo:
    # region: Utilities
    @staticmethod
    def get_statistics(
        data: np.typing.ArrayLike,
        is_sample: bool = False,
        return_z_score: bool = False,
    ) -> dict:
        """Compute descriptive statistics for array-like data.
        
        Args:
            data: Input data (array-like: list, tuple, np.ndarray, pd.Series, etc.)
            is_sample: If True, use sample std/variance (ddof=1).
        
        Returns:
            Dictionary of statistics.
        """
        # Convert to numpy array (handles most array-like inputs)
        arr = StatisticsInfo._validate_and_convert_array(data)
        
        if arr.size == 0:
            raise ValueError("Input must not be empty.")
        
        # Check for non-numeric values (e.g., strings)
        if np.isnan(arr).any():
            raise ValueError("Input contains non-numeric values.")
        
        ddof = 1 if is_sample else 0
        mean = np.mean(arr)
        std = np.std(arr, ddof=ddof)
        
        result = {
            'mean': float(mean),
            'std': float(std),
            'count': int(len(arr)),
            'variance': float(np.var(arr, ddof=ddof)),
            'cv': float(std / mean) if mean != 0 else np.nan,
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'q1': float(np.percentile(arr, 25)),
            'q2': float(np.percentile(arr, 50)),
            'q3': float(np.percentile(arr, 75)),
            'iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        }
        
        if return_z_score:
            result.update({'z_score': ((arr - mean) / std).tolist()})
            
        return result
        
    @staticmethod
    def get_z_critical(confidence: float) -> float:
        """Get critical z-value for a given confidence level."""
        return float(norm.ppf(1 - (1 - confidence)/2))
    
    @staticmethod
    def get_t_critical(confidence: float, df: int) -> float:
        """Get critical t-value for a given confidence level and degrees of freedom."""
        return float(t.ppf(1 - (1 - confidence)/2, df))
    
    @staticmethod
    def calculate_bin_width(
        data: np.typing.ArrayLike, 
        rule: Literal["sturges", "sqrt", "rice"] = "sturges"
    ) -> tuple[int, float]:
        """Calculate optimal histogram bin width and count using specified rule.
        
        Supports three common binning strategies:
        - Sturges' Rule (default): Ideal for normal distributions, smaller datasets
        - Square Root Rule: Simpler alternative for uniform distributions
        - Rice Rule: Better for larger datasets
        
        Args:
            data: Input data (array-like). Will be converted to numpy array.
            rule: Binning rule to use. One of:
                - "sturges": k = ceil(1 + 3.322*log10(n))
                - "sqrt": k = ceil(sqrt(n))
                - "rice": k = ceil(2*n^(1/3))
                Defaults to "sturges".
                
        Returns:
            Tuple containing:
            - num_bins (int): Number of bins (always rounded up to integer)
            - bin_width (float): Width of each bin (rounded to 2 decimal places)
            
        Raises:
            ValueError: If input data is empty or contains non-numeric values,
                or if invalid rule is specified.
        """ 
        arr = StatisticsInfo._validate_and_convert_array(data)
        n = len(arr)
        
        match rule:
            case "sturges":
                num_bins = math.ceil(1 + 3.322 * np.log10(n)) if n > 0 else 1
            case "sqrt":
                num_bins = math.ceil(np.sqrt(n))
            case "rice":
                num_bins = math.ceil(2 * (n ** (1/3)))
            case _:
                raise ValueError("Invalid rule. Use 'sturges', 'sqrt', or 'rice'.")
        
        bin_width = float(np.round((np.max(arr) - np.min(arr)) / num_bins, 2))
        return num_bins, bin_width
    
    @staticmethod
    def _validate_and_convert_array(data: np.typing.ArrayLike, min_length: int = 1) -> np.ndarray:
        """Convert ArrayLike input to numpy array with validation.
        
        Args:
            data: Input data (array-like).
            
        Returns:
            Validated numpy array.
            
        Raises:
            ValueError: If conversion fails or data is empty/non-numeric.
        """
        try:
            arr = np.asarray(data, dtype=np.float64)
            if arr.size == 0:
                raise ValueError("Input must not be empty.")
            if np.isnan(arr).any():
                raise ValueError("Input contains non-numeric values.")
            if len(arr) < min_length:
                raise ValueError(f"Input must have at least {min_length} elements.")
            return arr
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid input data: {str(e)}") from e
    # endregion: Utilities
    
    # region: Mean related
    @staticmethod
    def calculate_mean_sample_size(
        confidence: float,
        error: float,
        std_dev: float,
        population_size: int | None = None
    ) -> int:
        """
        Calculate the required sample size for a given confidence level and margin of error.
        
        Args:
            confidence: confidence level
            error: margin of error
            std_dev: population standard deviation
            population_size: population size (optional, for finite population correction)
        
        Returns:
            Required sample size (rounded up to nearest integer)
        """
        if std_dev is None:
            raise ValueError("Population standard deviation (std_dev) must be specified for mean estimation")
        
        # Get critical z-value
        critical_value = StatisticsInfo.get_z_critical(confidence)
        
        if population_size is None:
            # Infinite population formula for mean
            n = ((critical_value * std_dev) / error)**2
        else:
            # Finite population formula for mean
            numerator = population_size * std_dev**2 * critical_value**2
            denominator = ((population_size - 1) * error**2) + (std_dev**2 * critical_value**2)
            n = numerator / denominator
        
        return math.ceil(n)
    
    @staticmethod
    def calculate_mean_error(
        confidence: float,
        data_size: int,
        std_dev: float,
        is_sample_std: bool = False,
    ) -> float:
        """Calculate margin of error for estimated means.
    
        Args:
            confidence: Confidence level.
            data_size: Data size.
            std_dev: Standard Deviation value.
            is_sample_std: Check Standard Deviation. If True use t-distribution; else normal-distribution.
        
        Returns:
            Margin of error.
        """
        # Validation step
        if not 0 < confidence < 1:
            raise ValueError("Confidence must be between 0 and 1")
        if data_size <= 0:
            raise ValueError("Data size must be positive")
        
        # Validate mean-specific parameters
        if std_dev is None:
            raise ValueError("Standard deviation is required for mean calculations")
        
        if std_dev <= 0:
            raise ValueError("Standard deviation must be positive")
        
        # Choose appropriate distribution
        if is_sample_std:
            critical_value = StatisticsInfo.get_t_critical(confidence, df=data_size - 1)
        else:
            critical_value = StatisticsInfo.get_z_critical(confidence)
        
        standard_error = std_dev / np.sqrt(data_size)
        return float(critical_value * standard_error)
    
    @staticmethod
    def calculate_mean_pvalue(
        data_size: int,
        sample_mean: float,
        std_dev: float,
        significance: float,
        hypothesis_mean: float,
        alternative: Literal["left", "right", "two-sided"],
        is_sample_std: bool,
        print_evaluation: bool = False,
    ):
        std_error = std_dev / np.sqrt(data_size)
        stat = (sample_mean - hypothesis_mean) / std_error
        
        if is_sample_std:
            # T-test
            gl = data_size - 1
            cdf = partial(t.cdf, df=gl)
        else:
            # Z-test
            cdf = norm.cdf
        
        match alternative:
            case 'left':
                p_value = cdf(stat)
            case 'right':
                p_value = 1 - cdf(stat)
            case 'two-sided':
                p_value = 2 * (1 - cdf(abs(stat)))
            case _:
                raise ValueError("Invalidade alternative: must be 'left', 'right' or 'two-sided'")
        
        if print_evaluation:
            decision = "Reject H0" if p_value < significance else "Accept H0"
            print(f"{'t' if is_sample_std else 'z'}_stat={stat:.4f}")
            print(f"{p_value=:.5f}")
            print(f"{decision=}")
        
        return p_value
    # endregion: Mean related
    
    # region: Proportion related
    @staticmethod
    def calculate_proportion_sample_size(
        confidence: float,
        error: float,
        p: float,
        population_size: int | None = None 
    ) -> int:
        """
        Calculate the required sample size for a given confidence level and margin of error.
        
        Args:
            confidence: confidence level
            error: margin of error
            p: estimated proportion
            population_size: population size (optional, for finite population correction)
        
        Returns:
            Required sample size (rounded up to nearest integer)
        """
        if p is None:
            raise ValueError("Estimated proportion (p) must be specified")
        
        # Get critical z-value
        critical_value = StatisticsInfo.get_z_critical(confidence)
            
        if population_size is None:
            # Infinite population formula for proportion
            n = (critical_value**2 * p * (1 - p)) / (error**2)
        else:
            # Finite population formula for proportion
            numerator = population_size * p * (1 - p) * critical_value**2
            denominator = (p * (1 - p) * critical_value**2) + ((population_size - 1) * error**2)
            n = numerator / denominator
        
        return math.ceil(n)
    
    @staticmethod
    def calculate_proportion_error( 
        confidence: float,
        data_size: int,
        p: float
    ) -> float:
        """Calculate margin of error for estimated proportions.
    
        Args:
            confidence: Confidence level
            data_size: Data size
            p: Estimated proportion
        
        Returns:
            Margin of error.
        """
        # Validation step
        if not 0 < confidence < 1:
            raise ValueError("Confidence must be between 0 and 1")
        if data_size <= 0:
            raise ValueError("Data size must be positive")
        
        # Validate proportion-specific parameters
        if p is None:
            raise ValueError("Proportion 'p' is required for proportion calculations")
        
        if (p <= 0) or (p >= 1):
            raise ValueError("Proportion 'p' must be between 0 and 1")
        
        # Check normal approximation conditions
        if data_size * p < 5 or data_size * (1 - p) < 5:
            raise ValueError(
                "Normal approximation not valid - np and n(1-p) should be ≥ 5. "
                f"Got np={data_size*p}, n(1-p)={data_size*(1-p)}"
            )
        
        critical_value = StatisticsInfo.get_z_critical(confidence)
        standard_error = np.sqrt(p * (1 - p) / data_size)
        return float(critical_value * standard_error)
    
    @staticmethod
    def calculate_proportion_pvalue(
        data_size: int,
        sample_proportion: float,
        significance: float,
        hypothesis_proportion: float,
        alternative: Literal["left", "right", "two-sided"],
        print_evaluation: bool = False,
    ):
        std_error = np.sqrt(hypothesis_proportion*(1-hypothesis_proportion)/data_size)
        stat = (sample_proportion - hypothesis_proportion) / std_error
        
        cdf = norm.cdf
        
        match alternative:
            case 'left':
                p_value = cdf(stat)
            case 'right':
                p_value = 1 - cdf(stat)
            case 'two-sided':
                p_value = 2 * (1 - cdf(abs(stat)))
            case _:
                raise ValueError("Invalidade alternative: must be 'left', 'right' or 'two-sided'")
        
        if print_evaluation:
            decision = "Reject H0" if p_value < significance else "Accept H0"
            print(f"z_stat={stat:.4f}")
            print(f"{p_value=:.5f}")
            print(f"{decision=}")
        
        return p_value
    # endregion: Proportion related
    
    @staticmethod
    def get_outlier_limit(df: pd.Series) -> tuple:
        # IQR - InterQuartile Range method
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)

        inferior_limit = Q1 - 1.5 * (Q3 - Q1)
        superior_limit = Q3 + 1.5 * (Q3 - Q1)

        return inferior_limit, superior_limit
    
    @staticmethod
    def detect_outliers(series: pd.Series, method: str = 'iqr', threshold: float | None = None) -> pd.Series:
        if method not in ['iqr','zscore', 'mad']:
            raise ValueError("Invalid method. Use 'iqr', 'zscore', or 'mad'.")
        
        if method == 'iqr':
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            thresh = 1.5 if threshold is None else threshold
            lower = q1 - thresh * iqr
            upper = q3 + thresh * iqr
            return (series < lower) | (series > upper)
        
        elif method == 'zscore':
            thresh = 3 if threshold is None else threshold
            z_scores = zscore(series)
            return np.abs(z_scores) > thresh
        
        elif method == 'mad':
            thresh = 3.5 if threshold is None else threshold
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z = 0.6745 * (series - median) / mad
            return np.abs(modified_z) > thresh