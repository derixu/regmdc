import numpy as np
import cvxpy as cp

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
