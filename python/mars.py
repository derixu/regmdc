import numpy as np
import pandas as pd
import cvxpy as cp

def compute_hinge(x):
    return np.maximum(0, x)


def scale_back_matrix_entry(entry, max_val, min_val, digits=4):
    return format((max_val - min_val) * entry + min_val, f'.{digits}g')


def get_unique_column_entries(X_design, method='tcmars', number_of_bins=None):
    '''
    Currently only supports tcmars
    
    X_design: n x d numPy array or pandas DataFrame containing n datapoints with d features
    number_of_bins: length d list or numPy array containing bin counts (ints) for each column of X_design
    
    More type restrictive than R version 
        - number_of_bins cannot be an integer, and incorrect lengths of number_of_bins is not handled
        - if an element of number_of_bins is not an integer, fallback to default behavior (no bins).
    '''
    
    if isinstance(X_design, pd.DataFrame):
        X_design = X_design.values
    
    if method == 'tcmars':
        if number_of_bins is None:
            # list comprehension may be inefficient
            # np.unique sorts by default
            return [np.unique(np.append(0, X_design[:, i]))[:-1] for i in range(X_design.shape[1])]
        else:
            # for loop may be inefficient
            column_unique = []
            for i in range(X_design.shape[1]):
                N = number_of_bins[i]
                if not isinstance(N, (int)):
                    column_unique.append(np.unique(np.append(0, X_design[:, i]))[:-1])
                else:
                    column_unique.append(np.array(np.arange(N))/N)
            return column_unique
    else:
        raise ValueError("method must be tcmars")
    

def get_lasso_matrix_tcmars(X_eval,
                        X_design,
                        max_vals,
                        min_vals,
                        s=2,
                        number_of_bins=None,
                        concave_covariates=None,
                        convex_covariates=None,
                        variation_constrained_covariates=None,
                        extra_linear_covariates=None,
                        is_included_basis=None,
                        colnames=None
                        ):
    '''
    Solve for the matrix for the LASSO problem. This function only deals with NumPy based objects.
    
    X_eval: 2D numPy Array, each row corresponds to an individual evaluation point at which basis 
    functions are computed.
    
    X_design: 2D numPy Array, each row corresponds to an individual data point, from which 
    basis functions are constructed from this matrix, using values from X_design as knots
    
    max_vals: 1D numPy Array, array of maximal covariate values
    
    min_vals: 1D numPy Array, array of minimal covariate values
    
    s: int, maximum degree of interaction between covariates allowed
    
    number_of_bins: None OR 1D numPy Array, length equal to number of covariates, each element corresponds to
    the number of bins used for approximate method for creating hinges on a specific covariate
    
    concave_covariates: None OR 1D numPy Array, each element is the column index corresponding to a covariate 
    that will be restricted to be concave
    
    convex_covariates=None: None OR 1D numPy Array, each element is the column index corresponding to a covariate 
    that will be restricted to be convex
    
    variation_constrained_covariates: None OR 1D numPy Array, each element is the column index corresponding 
    to a covariate that will be variation restricted
    
    extra_linear_covariates: None OR 1D numPy Array, each element is the column index corresponding to a covariate 
    that will be kept linear and not included in any hinges
    
    is_included_basis: None OR 1D numPy Array, length equal to the number of columns in the final LASSO matrix, 
    each element corresponding to an indicator for whether or not that column is included in the LASSO problem. 
    Primarily used when creating a new matrix to predict on using a trained MARS-via-LASSO model
    
    colnames: None or 1D numPy Array, length equal to the number of covariates in X_eval and X_design. 
    Used to label columns of the returned LASSO matrix. If not passed in, default colnames will be x1, x2, x3, ..., etc.
    '''
    
    
    # We only drop bases if we have both concave and convex constraints on our bases
    is_basis_drop_possible = concave_covariates is not None and convex_covariates is not None
    
    # Create indicator arrays for whether or not certain covariates have constraints on them
    is_convex_covariate = np.zeros(X_design.shape[1]).astype(bool)
    if convex_covariates is not None:
        is_convex_covariate[convex_covariates] = True
    
    is_concave_covariate = np.zeros(X_design.shape[1]).astype(bool)
    if is_concave_covariate is not None:
        is_concave_covariate[convex_covariates] = True
    
    is_variation_constrained_covariate = np.zeros(X_design.shape[1]).astype(bool)
    if is_variation_constrained_covariate is not None:
        is_variation_constrained_covariate[convex_covariates] = True
    
    # Create dummy colnames if colnames not provided
    if colnames is None:
        colnames = np.array([f"x{i + 1}" for i in range(X_eval.shape[1])])
    
    # Remove linear covariates from matrix. Add back at the end
    if extra_linear_covariates is not None:
        
        # Hold onto extra linear features and scale factors for our final LASSO matrix
        lasso_matrix_extra_linear = X_eval[:, extra_linear_covariates]
        basis_scale_factors_extra_linear = max_vals[extra_linear_covariates] - min_vals[extra_linear_covariates]
        linear_colnames = colnames[extra_linear_covariates]
        
        # Remove extra linear features from hinge/basis procedures
        keep_indices = [
            i for i in range(X_design.shape[1]) if i not in extra_linear_covariates
        ]

        X_eval = X_eval[:, keep_indices]
        X_design = X_design[:, keep_indices]
        max_vals = max_vals[keep_indices]
        min_vals = min_vals[keep_indices]
        if number_of_bins is not None: number_of_bins = number_of_bins[keep_indices]
        is_convex_covariate = is_convex_covariate[keep_indices]
        is_concave_covariate = is_concave_covariate[keep_indices]
        is_variation_constrained_covariate = is_variation_constrained_covariate[keep_indices]
        colnames = colnames[keep_indices]
        
    else:
        basis_scale_factors_extra_linear = None
    
    # Retrieve unique entries of each column and check for columns with only zero values
    d = X_design.shape[1]
    unique_entries = get_unique_column_entries(X_design, 'tcmars', number_of_bins)
    for col_idx in range(d):
        if len(unique_entries[col_idx]) == 0:
            raise ValueError(f"All the values of \'column {col_idx}\' are zero. Please remove that variable.")
    
    # Retrieve lasso matrix unlabeled
    lasso_matrix_rows = []
    for row_idx in range(X_eval.shape[0]):
        
        # Compute hinges for each row
        
        '''
        Each element of the hinges list holds an array corresponding to a covariate containing the constant term 
        and each unique hinge on that covariate.
        
        Each element of the hinges_order list holds an array corresponding to a covariate containing the orders 
        of each element in the corresponding hinges array (0 for constant, 1 for all others)
        '''
        hinges = []
        hinges_orders = []
        
        for col_idx in range(d):
            col_hinges = np.concatenate(([1], compute_hinge(X_eval[row_idx, col_idx] - unique_entries[col_idx])))
            hinges.append(col_hinges)
            
            col_hinges_orders = np.concatenate(([0], np.ones(len(unique_entries[col_idx]), dtype=int)))
            hinges_orders.append(col_hinges_orders)
        
        if is_basis_drop_possible:
            '''
            Each element of these lists holds an array corresponding to one covariate.
            
            The array holds an indicator for each unique hinge on that covariate, where a True value represents a 
            constraint. The first two hinges (constant, linear) automatically have no constraint
            '''
            is_positive_hinge = []
            is_negative_hinge = []
            is_variation_constrained_hinge = []
            
            for col_idx in range(d):
                num_unique_col = len(unique_entries[col_idx])
                
                if is_convex_covariate[col_idx]:
                    pos_hinges = np.concatenate(([False, False], np.ones(num_unique_col-1, dtype=bool)))
                else:
                    pos_hinges = np.zeros(num_unique_col + 1, dtype=bool)
                
                if is_concave_covariate[col_idx]:
                    neg_hinges = np.concatenate(([False, False], np.ones(num_unique_col-1, dtype=bool)))
                else:
                    neg_hinges = np.zeros(num_unique_col + 1, dtype=bool)
                    
                if is_variation_constrained_covariate[col_idx]:
                    var_hinges = np.concatenate(([False, False], np.ones(num_unique_col-1, dtype=bool)))
                else:
                    var_hinges = np.zeros(num_unique_col + 1, dtype=bool)
                
                is_positive_hinge.append(pos_hinges)
                is_negative_hinge.append(neg_hinges)
                is_variation_constrained_hinge.append(var_hinges)
        
        # Initialize basis functions
        basis = hinges[0]
        basis_orders = hinges_orders[0]
        
        if is_basis_drop_possible:
            is_positive_basis = is_positive_hinge[0]
            is_negative_basis = is_negative_hinge[0]
            is_variation_constrained_basis = is_variation_constrained_hinge[0]
        
        # Collect basis functions that meet interaction order s (so we can remove them from the pool of bases)
        full_order_basis = []
        
        if d >= 2:
            for k in range(1, d):
                # Collect and remove basis functions that are already at maximum allowed s
                is_full_order = basis_orders == s
                full_order_basis.extend(basis[is_full_order])
                
                basis = basis[~is_full_order]
                basis_orders = basis_orders[~is_full_order]
                
                # Construct new bases functions using old bases and new covariate k, each new basis = old basis * new hinge
                basis = np.outer(basis, hinges[k]).flatten()
                basis_orders = np.outer(basis_orders, hinges_orders[k]).flatten()
                
                # If we have constraints that may allow us to drop basis, check for possible cases
                if is_basis_drop_possible:
                    
                    # First, remove basis functions that were at max allowed s from constraint indicator arrays
                    is_positive_basis = is_positive_basis[~is_full_order]
                    is_negative_basis = is_negative_basis[~is_full_order]
                    is_variation_constrained_basis = is_variation_constrained_basis[~is_full_order]
                    
                    # If either old basis or new hinge multiplied in have constraint, new basis has constraint
                    is_positive_basis = np.logical_or.outer(
                        is_positive_basis, is_positive_hinge[k]
                    ).flatten()
                    
                    is_negative_basis = np.logical_or.outer(
                        is_negative_basis, is_negative_hinge[k]
                    ).flatten()
                    
                    is_variation_constrained_basis = np.logical_or.outer(
                        is_variation_constrained_basis,
                        is_variation_constrained_hinge[k],
                    ).flatten()
                    
                    # Remove bases that both require coefficients to be positive and negative
                    is_nonzero = ~(is_positive_basis & is_negative_basis)
                    basis = basis[is_nonzero]
                    basis_orders = basis_orders[is_nonzero]
                    is_positive_basis = is_positive_basis[is_nonzero]
                    is_negative_basis = is_negative_basis[is_nonzero]
                    is_variation_constrained_basis = is_variation_constrained_basis[is_nonzero]
                    
        # After all iterations, our final basis is composed of our remaining non full order bases and our stored full-order bases
        basis = np.concatenate((basis, full_order_basis))
        lasso_matrix_rows.append(basis)
    
    # After assembling each row, combine for complete matrix
    lasso_matrix = np.vstack(lasso_matrix_rows)
    
    # Create names for each unique hinge   
    hinges_names = []
    hinges_scale_factors = []
    
    for col_idx in range(d):
        
        # Construct all names for each hinge on a covariate, append to our list of hinge names
        names = [
            f"H({colnames[col_idx]}-{scale_back_matrix_entry(entry, max_vals[col_idx], min_vals[col_idx], digits=4)})"
            for entry in unique_entries[col_idx]
        ]
        names = np.array([""] + names) # add constant term label
        hinges_names.append(names)
        
        scale_factors = np.concatenate(([1.0], np.full(len(unique_entries[col_idx]), 1.0 / (max_vals[col_idx] - min_vals[col_idx]))))
        hinges_scale_factors.append(scale_factors)
    
    '''
    Create names for bases 
    
    For hinges_orders, is_positive_hinge, is_negative_hinge, is_variation_constrained_hinge, reuse from previous section on computing bases)
    Also track whether or not a basis positive, negative, or variation constrained, and the scale factors
    '''
    
    # Running lists tracking names, orders, constraints, scale factors
    basis_names = hinges_names[0]
    basis_orders = hinges_orders[0]
    is_positive_basis = is_positive_hinge[0]
    is_negative_basis = is_negative_hinge[0]
    is_variation_constrained_basis = is_variation_constrained_hinge[0]
    basis_scale_factors = hinges_scale_factors[0]
    
    # Lists where we store names, orders, constraints, scale factors, of already full order bases
    full_order_basis_names = []
    full_order_positive = []
    full_order_negative = []
    full_order_variation_constrained = []
    full_order_scale_factors = []
    
    for k in range(1, d):
        # Check for bases that are full order (track indices)
        is_full_order = (basis_orders == s)
        
        # Update full order lists
        full_order_basis_names.extend(np.array(basis_names)[is_full_order])
        full_order_positive.extend(np.array(is_positive_basis)[is_full_order])
        full_order_negative.extend(np.array(is_negative_basis)[is_full_order])
        full_order_variation_constrained.extend(np.array(is_variation_constrained_basis)[is_full_order])
        full_order_scale_factors.extend(np.array(basis_scale_factors)[is_full_order])
        
        # Remove full order bases from running lists
        basis_names = basis_names[~is_full_order]
        basis_orders = basis_orders[~is_full_order]
        is_positive_basis = is_positive_basis[~is_full_order]
        is_negative_basis = is_negative_basis[~is_full_order]
        is_variation_constrained_basis = is_variation_constrained_basis[~is_full_order]
        basis_scale_factors = basis_scale_factors[~is_full_order]
    
        # Update lists with combinations with new covariate
        
        # Note: .astype('object') allows us to add strings by restricting them to the same type (dynamically length strings) instead of '<U15', '<U20', etc. But it may cost efficiency.
        basis_names = np.add.outer(basis_names.astype('object'), hinges_names[k].astype('object')).flatten()
        basis_orders = np.add.outer(basis_orders, hinges_orders[k]).flatten()
        is_positive_basis = np.logical_or.outer(is_positive_basis, is_positive_hinge[k]).flatten()
        is_negative_basis = np.logical_or.outer(is_negative_basis, is_negative_hinge[k]).flatten()
        is_variation_constrained_basis = np.logical_or.outer(
            is_variation_constrained_basis, is_variation_constrained_hinge[k]
        ).flatten()
        basis_scale_factors = np.multiply.outer(basis_scale_factors, hinges_scale_factors[k]).flatten()
        
        # Drop basis that violate constraints
        if is_basis_drop_possible:
            is_nonzero = ~(is_positive_basis & is_negative_basis)
            basis_names = basis_names[is_nonzero]
            basis_orders = basis_orders[is_nonzero]
            is_positive_basis = is_positive_basis[is_nonzero]
            is_negative_basis = is_negative_basis[is_nonzero]
            is_variation_constrained_basis = is_variation_constrained_basis[is_nonzero]
            basis_scale_factors = basis_scale_factors[is_nonzero]
        
    # Combine full order lists with final running lists
    basis_names = np.concatenate((basis_names, full_order_basis_names))
    is_positive_basis = np.concatenate((is_positive_basis, full_order_positive))
    is_negative_basis = np.concatenate((is_negative_basis, full_order_negative))
    is_variation_constrained_basis = np.concatenate(
        (is_variation_constrained_basis, full_order_variation_constrained)
    )
    basis_scale_factors = np.concatenate((basis_scale_factors, full_order_scale_factors))
    
    # Substitute the empty string for constant term for the intercept label
    basis_names[0] = "(Intercept)"
    
    # Handle extra linear covariates
    if extra_linear_covariates is not None:
        lasso_matrix = np.hstack((lasso_matrix, lasso_matrix_extra_linear))
        is_positive_basis = np.concatenate(
            (is_positive_basis, np.zeros(len(extra_linear_covariates), dtype=bool))
        )
        is_negative_basis = np.concatenate(
            (is_negative_basis, np.zeros(len(extra_linear_covariates), dtype=bool))
        )
        is_variation_constrained_basis = np.concatenate(
            (is_variation_constrained_basis, np.zeros(len(extra_linear_covariates), dtype=bool))
        )
        basis_scale_factors = np.concatenate((basis_scale_factors, basis_scale_factors_extra_linear))
        
        basis_names = np.concatenate((basis_names, linear_colnames))
    
    # Determine which basis functions to include, if provided use is_included_basis, else include all non-zero columns
    if is_included_basis is None:
        is_included_basis = np.any(lasso_matrix != 0, axis=0)
    else:
        if len(is_included_basis) != lasso_matrix.shape[1]:
            raise ValueError('`length(is_included_basis)` must be equal to the number of columns of the LASSO matrix.')
    
    # Include only specified columns
    included_indices = np.where(is_included_basis)[0]
    
    lasso_matrix = lasso_matrix[:, included_indices]
    is_positive_basis = is_positive_basis[included_indices]
    is_negative_basis = is_negative_basis[included_indices]
    is_variation_constrained_basis = is_variation_constrained_basis[included_indices]
    basis_scale_factors = basis_scale_factors[included_indices]
    basis_names = basis_names[included_indices]
    
    return {
        'lasso_matrix': lasso_matrix,
        'colnames': basis_names,
        'is_positive_basis': is_positive_basis,
        'is_negative_basis': is_negative_basis,
        'is_variation_constrained_basis': is_variation_constrained_basis,
        'basis_scale_factors': basis_scale_factors,
        'is_included_basis': is_included_basis
    }
        

def solve_constrained_lasso(y, M, V=np.inf,
                        is_sum_constrained_component=None,
                        is_positive_component=None,
                        is_negative_component=None):
    n, p = M.shape

    if is_sum_constrained_component is None:
        is_sum_constrained_component = np.full(p, True, dtype=bool)
    if is_positive_component is None:
        is_positive_component = np.full(p, False, dtype=bool)
    if is_negative_component is None:
        is_negative_component = np.full(p, False, dtype=bool)

    # Define cvxpy variable
    x = cp.Variable(p)

    # Define objective function
    objective = cp.Minimize(cp.sum_squares(y - M @ x))

    # Begin composing list of constraints
    constraints = []

    # Sign constraints for x
    for i in range(p):
        if is_positive_component[i]:
            constraints.append(x[i] >= 0)
        if is_negative_component[i]:
            constraints.append(x[i] <= 0)

    # Regularization constraint: sum |x_i| <= V if V is finite
    if np.isfinite(V):
        indices = np.where(is_sum_constrained_component)[0]
        constraints.append(cp.norm1(x[indices]) <= V)

    # Create problem
    prob = cp.Problem(objective, constraints)

    # Solve problem
    prob.solve(solver=cp.MOSEK)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError("Optimization did not converge.")

    x_value = x.value

    return x_value
