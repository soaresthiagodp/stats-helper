import math
from functools import partial
from typing import Literal

import pandas as pd
import numpy as np
from scipy.stats import norm, f, t, shapiro, kstest, anderson, zscore, ttest_ind, ttest_rel

from statshelper.tools.validators import validate_and_convert_array


# Parametric Class #TODO: Non-parametric
class Hypothesis:
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
    ) -> float:
        """Perform a one-sample hypothesis test for means (Z-test or t-test).
    
        Conducts either:
        - Z-test when population standard deviation is known (is_sample_std=False)
        - t-test when using sample standard deviation (is_sample_std=True)

        Args:
            data_size: Sample size (n).
            sample_mean: Observed sample mean (data_1).
            std_dev: Standard deviation (sigma if population, s if sample).
            significance: Significance level (alpha) for the test.
            hypothesis_mean: Hypothesized population mean (mu0).
            alternative: Alternative hypothesis type:
                - 'left': data_1 < mu0 (one-tailed)
                - 'right': data_1 > mu0 (one-tailed)
                - 'two-sided': data_1 != mu0 (two-tailed)
            is_sample_std: Whether std_dev is from sample (t-test if True else z-test)
            print_evaluation: If True, prints test statistics and decision.

        Returns:
            p-value for the specified alternative hypothesis.

        Raises:
            ValueError: If invalid alternative hypothesis is specified
        """
        
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
                raise ValueError("Invalid alternative: must be 'left', 'right' or 'two-sided'")
        
        if print_evaluation:
            decision = "Reject H0" if p_value < significance else "Accept H0"
            print(f"{'t' if is_sample_std else 'z'}_stat={stat:.4f}")
            print(f"{p_value=:.5f}")
            print(f"{significance=:.5f}")
            print(f"{decision=}")
        
        return float(p_value)

    @staticmethod
    def calculate_proportion_pvalue(
        data_size: int,
        sample_proportion: float,
        significance: float,
        hypothesis_proportion: float,
        alternative: Literal["left", "right", "two-sided"],
        print_evaluation: bool = False,
    ) -> float:
        """Perform a one-sample Z-test for proportions using normal approximation.
    
        Tests whether the observed sample proportion differs significantly from
        a hypothesized population proportion. Uses the standard normal distribution
        (Z-test) which is valid when np ≥ 5 and n(1-p) ≥ 5.

        Args:
            data_size: Sample size (n).
            sample_proportion: Observed proportion in sample (p̂).
            significance: Significance level (alpha) for hypothesis test.
            hypothesis_proportion: Hypothesized population proportion (p0).
            alternative: Type of alternative hypothesis:
                - 'left': p < p0 (one-tailed)
                - 'right': p > p0 (one-tailed)
                - 'two-sided': pp != p0 (two-tailed)
            print_evaluation: If True, prints test statistics and decision.

        Returns:
            p-value for the test.

        Raises:
            ValueError: If invalid alternative hypothesis is provided
        """
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
                raise ValueError("Invalid alternative: must be 'left', 'right' or 'two-sided'")
        
        if print_evaluation:
            decision = "Reject H0" if p_value < significance else "Accept H0"
            print(f"z_stat={stat:.4f}")
            print(f"{p_value=:.5f}")
            print(f"{significance=:.5f}")
            print(f"{decision=}")
        
        return float(p_value)

    @staticmethod
    def compare_variances_f_test(
        variance_1: float,
        variance_2: float,
        size_1: int,
        size_2: int,
        significance: float,
        alternative: Literal["left", "right", "two-sided"],
        print_evaluation: bool = False,
    ) -> float:
        """Perform a F-test for equality of variances.
        
        Uses F-distribution where F = S1²/S2² (with S1² > S2²).

        Args:
            variance_1: Variance of sample 1.
            variance_2: Variance of sample 2.
            size_1: Sample size of group 1.
            size_2: Sample size of group 2.
            significance: Significance level (alpha).
            print_evaluation: If True, prints test statistics and decision.

        Returns:
            Two-tailed p-value for the F-test.

        Raises:
            ValueError: If sample variances are non-positive or sample sizes < 2.

        Notes:
            - For non-normal data, consider Levene's test instead.
            - Sensitive to normality assumption.
        """
        # TODO: Levene Test
        # Validate inputs
        if variance_1 <= 0 or variance_2 <= 0:
            raise ValueError("Sample variances must be positive")
        if size_1 < 2 or size_2 < 2:
            raise ValueError("Sample sizes must be >=2 2")

        f_stat = variance_1 / variance_2
        df_num = size_1 - 1
        df_den = size_2 - 1

        # TODO: Study about the swap logic and implement if necessary
        # if variance_1 >= variance_2:
        #     f_stat = variance_1 / variance_2
        #     df_num = size_1 - 1
        #     df_den = size_2 - 1
        # else:
            # f_stat = variance_2 / variance_1
            # df_num = size_2 - 1
            # df_den = size_1 - 1

        match alternative:
            case "left":
                # TODO: Verify if this is the correct approach
                p_value = f.cdf(f_stat, df_num, df_den)
            case "right":
                # TODO: Verify if this is the correct approach
                p_value = f.sf(f_stat, df_num, df_den)
            case "two-sided":
                if f_stat > 1:
                    p_value = 2 * f.sf(f_stat, df_num, df_den)
                else:
                    p_value = 2 * f.cdf(f_stat, df_num, df_den)
            case _:
                raise ValueError("Invalid alternative: must be 'left', 'right' or 'two-sided'")

        if print_evaluation:
            decision = "Reject H0" if p_value < significance else "Accept H0"
            print(f"F_stat={f_stat:.4f} df_num={df_num} df_den={df_den}")
            print(f"{p_value=:.5f}")
            print(f"{significance=:.5f}")
            print(f"{decision=}")

        return float(p_value)
    
    @staticmethod
    def compare_means_t_test(
        data_1: np.typing.ArrayLike,
        data_2: np.typing.ArrayLike,
        significance: float,
        alternative: Literal["two-sided", "left", "right"] = 'two-sided',
        paired: bool = False,
        print_evaluation: bool = False,
    ):
        """Perform a t-test to compare means between two samples

        Args:
            data_1 : First sample data array
            data_2 : Second sample data array
            significance : Significance level
            alternative : Literal["two-sided", "left", "right"]
                Type of alternative hypothesis:
                - "two-sided": means are not equal (default)
                - "left": mean of data_1 is less than mean of data_2
                - "right": mean of data_1 is greater than mean of data_2
            paired : bool
                Whether samples are paired/matched (e.g., before-after measurements).
                If True, performs a paired t-test (dependent samples).
            print_evaluation : bool, optional
                If True, prints test details including t-statistic, p-value, and decision.
                Default is False.

        Returns:
            float: The calculated p-value for the specified alternative hypothesis
        """

        arr_1 = validate_and_convert_array(data_1)
        arr_2 = validate_and_convert_array(data_2)

        f_test_pvalue = Hypothesis.compare_variances_f_test(
            variance_1=np.var(arr_1),
            variance_2=np.var(arr_2),
            size_1=len(arr_1),
            size_2=len(arr_2),
            significance=significance,
            alternative=alternative,
        )
        equal_var = False if f_test_pvalue < significance else True

        if paired:
            t_stat, p_value = ttest_rel(arr_1, arr_2)
        else:
            t_stat, p_value = ttest_ind(arr_1, arr_2, equal_var=equal_var)
        
        match alternative:
            case 'left':
                p_value = p_value/2 if t_stat < 0 else 1 - p_value/2
            case 'right':
                p_value = p_value/2 if t_stat > 0 else 1 - p_value/2
            case 'two-sided':
                pass # Just pass t_stat and p_value directly
            case _:
                raise ValueError("Invalid alternative: must be 'left', 'right' or 'two-sided'")
                
        if print_evaluation:
            decision = "Reject H0" if p_value < significance else "Accept H0"
            print(f"{equal_var=}")
            print(f"{paired=}")
            print(f"t_stat={t_stat:.4f}")
            print(f"{p_value=:.5f}")
            print(f"{significance=:.5f}")
            print(f"{decision=}")

        return p_value

    @staticmethod
    def compare_proportions_z_test(
        success_count_1: int,
        success_count_2: int,
        size_1: int,
        size_2: int,
        significance: float,
        alternative: Literal["two-sided", "left", "right"],
        print_evaluation: bool = False,
    ) -> dict:
        """Perform a two-proportion z-test to compare success rates between two groups.

        Conducts a hypothesis test comparing proportions from two independent samples using
        normal approximation, it does not use Yates Correction

        Args:
            success_count_1: Number of successes in group 1
            success_count_2: Number of successes in group 2
            size_1: Total number of observations in group 1
            size_2: Total number of observations in group 2
            significance: Significance level (alpha) for hypothesis test
            alternative: Type of alternative hypothesis:
                "two-sided": test for difference in proportions (p1 != p2)
                "left": test if proportion 1 is less than proportion 2 (p1 < p2)
                "right": test if proportion 1 is greater than proportion 2 (p1 > p2)
            print_evaluation: If True, prints test statistics and decision

        Returns:
            float: The calculated p-value for the specified alternative hypothesis

        Raises:
            ValueError: If success counts are negative or sample sizes are non-positive
            ValueError: If alternative hypothesis is not one of the specified options
        """
        # TODO: Fisher Test for small sample

        # Validate inputs
        if success_count_1 < 0 or success_count_2 < 0:
            raise ValueError("Success counts cannot be negative")
        if size_1 <= 0 or size_2 <= 0:
            raise ValueError("Sample sizes must be positive")
        
        # Calculate proportions
        p1 = success_count_1 / size_1
        p2 = success_count_2 / size_2
        p_pooled = (success_count_1 + success_count_2) / (size_1 + size_2)
        
        # Standard error calculation (with/without continuity correction)
        # TODO: Yates continuity correction
        # adj = 0.5 * (1/size_1 + 1/size_2)
        # se = np.sqrt(p_pooled * (1 - p_pooled) * (1/size_1 + 1/size_2))
        # z_num = abs(p1 - p2) - adj

        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/size_1 + 1/size_2))
        z_num = abs(p1 - p2)
        
        z_stat = z_num / se
        
        # Calculate p-value based on alternative
        match alternative:
            case 'left':
                p_value = norm.cdf((p1 - p2)/se)
            case 'right':
                p_value = norm.sf((p1 - p2)/se)
            case 'two-sided':
                p_value = 2 * (1 - norm.cdf(z_stat))
            case _:
                raise ValueError("Invalid alternative: must be 'left', 'right' or 'two-sided'")
        
        if print_evaluation:
            decision = "Reject H0" if p_value < significance else "Accept H0"
            print(f"z_stat={z_stat:.4f}")
            print(f"{p_value=:.5f}")
            print(f"{significance=:.5f}")
            print(f"{decision=}")

        return float(p_value)

    @staticmethod
    def normality_tests(
        data: np.typing.ArrayLike
    ) -> dict:
        """Perform multiple normality tests on a dataset and return comprehensive results.
        
        Conducts three statistical tests to assess whether a sample comes from a normally
        distributed population:
        1. Shapiro-Wilk (best for small to medium samples)
        2. Kolmogorov-Smirnov (with sample parameters)
        3. Anderson-Darling (with critical values for multiple significance levels)

        Args:
            data: Input array-like data to test. Can be numpy array, list, or tuple of floats.
                Should contain at least 3 observations for Shapiro-Wilk and 5 for Anderson-Darling.

        Returns:
            Dictionary containing test results with structure:
            {
                'test_name': {
                    'statistic': float,             # Test normality, closer to 1 means normal
                    'p_value': float,               # Only for Shapiro-Wilk and KS
                    'critical_values': tuple,       # Only for Anderson-Darling
                    'significance_levels': tuple    # Only for Anderson-Darling
                }
            }

        Raises:
            ValueError: If input data contains less than 3 observations.
            TypeError: If input data contains non-numeric values.
        """
        arr = validate_and_convert_array(data, 3)

        results: dict = {}

        # Shapiro-Wilk (best for n <= 5000)
        stat, p = shapiro(arr)
        results['shapiro-wilk'] = {
            'statistic': float(stat),
            'p_value': float(p),
        }

        # Kolmogorov-Smirnov (with estimated parameters)
        mean, std = np.mean(arr), np.std(arr, ddof=1)
        stat, p = kstest(arr, 'norm', args=(mean, std))
        results['kolmogorov-smirnov'] = {
            'statistic': float(stat),
            'p_value': float(p),
        }

        # Anderson-Darling (with critical value comparison)
        ad_result = anderson(arr, dist='norm')
        results['anderson-darling'] = {
            'statistic': float(ad_result.statistic),
            'critical_values': tuple(ad_result.critical_values.tolist()),
            'significance_levels': tuple(ad_result.significance_level.tolist()),
        }

        return results


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
        arr = validate_and_convert_array(data)
        
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
        arr = validate_and_convert_array(data)
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