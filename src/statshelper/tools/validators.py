import numpy as np

def validate_and_convert_array(data: np.typing.ArrayLike, min_length: int = 1) -> np.ndarray:
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