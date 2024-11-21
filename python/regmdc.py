import numpy as np
import pandas as pd
from typing import Union


def regmdc(X_design: Union[np.ndarray, pd.DataFrame],
           y: np.ndarray,
           s: int,
           method: str,
           V: float = np.inf,
           threshold: float = 1e-6,
           is_scaled: bool = False,
           concave_covariates: np.ndarray = None,
           convex_covariates: np.ndarray = None,
           variation_constrained_covariates: np.ndarray = None,
           extra_linear_covariates: np.ndarray = None):
    
    # ======== ERROR HANDLING ================================================
    
    if not isinstance(X_design, (np.ndarray, pd.DataFrame)):
        raise TypeError('`X_design` must be a NumPy array or Pandas DataFrame.')
    
    if isinstance(X_design, np.ndarray):
        if len(X_design.shape) != 2:
            raise ValueError('`X_design` must be a matrix.')
    
        if not np.issubdtype(X_design.dtype, np.number):
            raise ValueError('Elements of `X_design` must have numeric values.')
        
        if X_design.size == 0:
            raise ValueError('`X_design` must not be empty.')

    elif isinstance(X_design, pd.DataFrame):
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_design.dtypes):
            raise ValueError('All columns in `X_design` must be numeric.')
        
        if X_design.empty:
            raise ValueError('`X_design` must not be empty.')


    if not isinstance(y, (np.ndarray, pd.Series)):
        raise TypeError('`y` must be a NumPy array or Pandas Series.')
    
    if isinstance(y, np.ndarray):
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError('Elements of `y` must have numeric values.')
        
        if any(np.isnan(y)):
            raise ValueError('Elements of `y` must not be NaN.')
        
        if len(y.shape) != 1:
            raise ValueError('`y` must be a vector.')
        
        if y.size == 0:
            raise ValueError('`y` must not be empty.')

        
    elif isinstance(y, pd.Series):
        pd.to_numeric(y, errors='raise')

        if any(pd.isna(y)):
            raise ValueError('Elements of `y` must not be NaN.')
    
    if len(y) != X_design.shape[0]:
        raise ValueError('`len(y)` must be equal to `X_design.shape[1]`.')

    if isinstance(s, int):
        raise TypeError('`s` must be an integer.')
    
    if s < 1 or s > X_design.shape[1]:
        raise ValueError('`s` must be at least 1 and at most `X_design.shape[1]`.')
    
    if method not in ['tc', 'mars', 'tcmars']:
        raise ValueError('`method` must be one of "tc", "mars", "tcmars".')
    
    if not isinstance(V, (int, float)):
        raise TypeError('`V` must be a numeric value.')
    
    if V <= 0:
        raise ValueError('`V` must be positive.')

    if not isinstance(threshold, (int, float)):
        raise TypeError('`threshold` must be a numeric value.')
    
    if threshold <= 0:
        raise ValueError('`threshold` must be positive.')
    
    if not isinstance(is_scaled, bool):
        raise TypeError('`is_scaled` must be True or False.')
    
    if is_scaled:
        if isinstance(X_design, np.ndarray):
            if np.min(X_design) < 0 or np.max(X_design) > 1:
                raise ValueError('If `is_scaled` is True, all entries of `X_design must be between 0 and 1.')
        elif isinstance(X_design, pd.DataFrame):
            if np.min(X_design.values) < 0 or np.max(X_design.values) > 1:
                raise ValueError('If `is_scaled` is True, all entries of `X_design must be between 0 and 1.')
    
    if number_of_bins is not None:
        if not isinstance(number_of_bins, (list, np.ndarray)):
            raise TypeError('`number_of_bins` must be a list or a NumPy Array.') # TODO: add support for pd.Series()
        
        if len(number_of_bins) != X_design.shape[1]:
            raise ValueError('`number_of_bins` must have the same length as the number of columns in X_design.')



        