from ortools.linear_solver import pywraplp
import numpy as np

def solver(criterion_data, preferences_ranking, criteria_bounds, alpha, delta=0.01, s_threshold=None):
    """ 
    Args:
        criterion_data = dict of criteria/attributes ; dict {action_id:[g1....gn]} 
        preference_ranking = list of action ids from best to worst 
        criteria bounds = list of (g_min, g_max) tuples
        alpha_segments = list of ints, number of grid points per criterion
        delta = minimum noticeable difference default 0.01
        s_threshold = list of monotonicity thresholds default at 0
    Returns:
        solver= solved LP model
        utilities= list of arrays [u_i(g_1), ..., u_i(g_alpha)]
        errors= dict {action: sigma_value}
        grids= list of arrays, grid points per criterion
    """
    
    solver = pywraplp.Solver.CreateSolver("GLOP")
    
    if not solver:
        raise RuntimeError("solver not available")
    n_criterion = len(criteria_bounds)
    actions = list(criterion_data.keys())
    if s_threshold is None:
        s_threshold = [0.0] * n_criterion
        
    #step 1
    grids = []
    for i in range (n_criterion):
        g_min, g_max = criteria_bounds[i]
        grid = np.linspace(g_min, g_max, alpha[i])
        grids.append(grid)    
    
    # step 2
    u = []
    for i in range(n_criterion):
        u_i = []
        for j in range(alpha[i]):
            var = solver.NumVar(0, 1.0, f'u_{i}_{j}')
            u_i.append(var)
        u.append(u_i)
    
    # sigma[a] = error for action a
    sigma = {a: solver.NumVar(0, solver.infinity(), f'sigma_{a}') 
             for a in actions}
    
    #step 3
    c_norm = solver.Constraint(1.0, 1.0, 'normalization')
    for i in range(n_criterion):
        c_norm.SetCoefficient(u[i][-1], 1.0)  # Last point = g_i*
    
    # Constraint (14): Monotonicity u_i(g^{j+1}) >= u_i(g^j) + s_i
    for i in range(n_criterion):
        for j in range(alpha[i] - 1):
            c_mono = solver.Constraint(s_threshold[i], solver.infinity(), f'mono_{i}_{j}')
            c_mono.SetCoefficient(u[i][j+1], 1.0)
            c_mono.SetCoefficient(u[i][j], -1.0)
    
    # Constraint (11): Preference U(a) - U(b) + σ(a) - σ(b) >= δ
    # Only adjacent ranks (transitivity)
    for rank_idx in range(len(preferences_ranking) - 1):
        a = preferences_ranking[rank_idx]
        b = preferences_ranking[rank_idx + 1]
        
        c_pref = solver.Constraint(delta, solver.infinity(), f'pref_{a}_{b}')
        
        # Add U(a) - U(b) using linear interpolation
        for i in range(n_criterion):
            g_a = criterion_data[a][i]
            g_b = criterion_data[b][i]
            
            # Interpolate for action a
            coeffs_a = _interpolate(g_a, grids[i])
            for j, coeff in enumerate(coeffs_a):
                if coeff > 0:
                    c_pref.SetCoefficient(u[i][j], coeff)
            
            # Interpolate for action b (negative)
            coeffs_b = _interpolate(g_b, grids[i])
            for j, coeff in enumerate(coeffs_b):
                if coeff > 0:
                    c_pref.SetCoefficient(u[i][j], -coeff)
        
        # Add σ(a) - σ(b)
        c_pref.SetCoefficient(sigma[a], 1.0)
        c_pref.SetCoefficient(sigma[b], -1.0)
    
# Step 4: Define objective min Σσ(a)
    objective = solver.Objective()
    for a in actions:
        objective.SetCoefficient(sigma[a], 1.0)
    objective.SetMinimization()
    
    # Step 5: Solve
    status = solver.Solve()
    
    if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        raise RuntimeError(f"Solver failed: {status}")
    
    # Extract solution
    utilities = [[var.solution_value() for var in u_i] for u_i in u]
    errors = {a: var.solution_value() for a, var in sigma.items()}
    
    return solver, utilities, errors, grids


def _interpolate(g_value, grid):
    """
    Linear interpolation: return coefficients for grid points.
    If g_value in [grid[j], grid[j+1]], returns array where:
    - coeff[j] = (grid[j+1] - g_value) / (grid[j+1] - grid[j])
    - coeff[j+1] = (g_value - grid[j]) / (grid[j+1] - grid[j])
    - all other coeffs = 0
    """
    coeffs = np.zeros(len(grid))
    
    # Find interval
    for j in range(len(grid) - 1):
        if grid[j] <= g_value <= grid[j+1]:
            delta = (g_value - grid[j]) / (grid[j+1] - grid[j])
            coeffs[j] = 1.0 - delta
            coeffs[j+1] = delta
            return coeffs
    
    # Edge cases
    if g_value <= grid[0]:
        coeffs[0] = 1.0
    elif g_value >= grid[-1]:
        coeffs[-1] = 1.0
    
    return coeffs


def evaluate_action(action_values, utilities, grids):
    """
    Calculate U(action) = Σu_i(g_i) using interpolation.
    """
    total = 0.0
    for i, g_val in enumerate(action_values):
        coeffs = _interpolate(g_val, grids[i])
        u_i = sum(c * u for c, u in zip(coeffs, utilities[i]))
        total += u_i
    return total


def calculate_kendall_tau(preference_ranking, criteria_data, utilities, grids):
    """
    Kendall's τ between original and predicted rankings.
    """
    # Calculate utilities for all actions
    U_dict = {a: evaluate_action(criteria_data[a], utilities, grids) 
              for a in criteria_data}
    
    # Predicted ranking (sorted by utility, descending)
    predicted = sorted(U_dict.keys(), key=lambda a: U_dict[a], reverse=True)
    
    # Count concordant/discordant pairs
    concordant = discordant = 0
    n = len(preference_ranking)
    
    for i in range(n):
        for j in range(i + 1, n):
            a_i, a_j = preference_ranking[i], preference_ranking[j]
            
            # Original: a_i better than a_j
            # Predicted: check if same
            pred_i = predicted.index(a_i)
            pred_j = predicted.index(a_j)
            
            if pred_i < pred_j:
                concordant += 1
            else:
                discordant += 1
    
    total_pairs = concordant + discordant
    tau = (concordant - discordant) / total_pairs if total_pairs > 0 else 0
    return tau


def print_results(utilities, errors, grids, criteria_data, preference_ranking):
    """
    Display results clearly.
    """
    print("=" * 60)
    print("UTA METHOD RESULTS")
    print("=" * 60)
    
    # Total error
    total_error = sum(errors.values())
    print(f"\nTotal Error F*: {total_error:.6f}")
    
    if total_error < 1e-6:
        print("✓ Perfect consistency achieved!")
    
    # Criterion weights (utility at max value)
    print("\nCriterion Weights:")
    for i, u_i in enumerate(utilities):
        weight = u_i[-1]  # u_i(g_i*)
        print(f"  Criterion {i+1}: {weight:.4f}")
    
    # Individual errors
    print("\nAction Errors:")
    for action in preference_ranking:
        print(f"  {action}: σ = {errors[action]:.6f}")
    
    # Kendall's tau
    tau = calculate_kendall_tau(preference_ranking, criteria_data, utilities, grids)
    print(f"\nKendall's τ: {tau:.4f}")
    
    # Predicted utilities
    print("\nPredicted Utilities:")
    U_values = [(a, evaluate_action(criteria_data[a], utilities, grids)) 
    for a in preference_ranking]
    for action, u_val in sorted(U_values, key=lambda x: x[1], reverse=True):
        rank = preference_ranking.index(action) + 1
        print(f"  {action}: U = {u_val:.4f} (original rank: {rank})")
        

if __name__ == "__main__":
    
    # Criterion values for 10 cars
    # [max_speed, consumption_town, consumption_120, horsepower, space, price]
    cars = {
        'Peugeot_505_GR':       [173, 11.4, 10.01, 10, 7.88, 49500],
        'Opel_Record_2000_LS':  [176, 12.3, 10.48, 11, 7.96, 46700],
        'Citroen_Visa_Super_E': [142, 8.2,  7.30,  5,  5.65, 32100],
        'VW_Golf_1300_GLS':     [148, 10.5, 9.61,  7,  6.15, 39150],
        'Citroen_CX_2400':      [178, 14.5, 11.05, 13, 8.06, 64700],
        'Mercedes_230':         [180, 13.6, 10.40, 13, 8.47, 75700],
        'BMW_520':              [182, 12.7, 12.26, 11, 7.81, 68593],
        'Volvo_244_DL':         [145, 14.3, 12.95, 11, 8.38, 55000],
        'Peugeot_104_ZS':       [161, 8.6,  8.42,  7,  5.11, 35200],
        'Citroen_Dyane':        [117, 7.2,  6.75,  3,  5.81, 24800],
    }
    
    # Subjective ranking (1 = best)
    ranking = [
        'Peugeot_505_GR',
        'Opel_Record_2000_LS',
        'Citroen_Visa_Super_E',
        'VW_Golf_1300_GLS',
        'Citroen_CX_2400',
        'Mercedes_230',
        'BMW_520',
        'Volvo_244_DL',
        'Peugeot_104_ZS',
        'Citroen_Dyane',
    ]
    
    # Criterion bounds (Table 4)
    bounds = [
        (110, 190),    # Max speed
        (7, 15),       # Consumption town
        (6, 13),       # Consumption 120
        (3, 13),       # Horsepower
        (5, 9),        # Space
        (20000, 80000) # Price
    ]
    
    # Number of grid points per criterion (Table 4)
    alphas = [5, 4, 4, 5, 4, 5]
    
    # Solve UTA
    print("Solving UTA model...")
    solver, utilities, errors, grids = solver(
        criterion_data=cars,
        preferences_ranking=ranking,
        criteria_bounds=bounds,
        alpha=alphas,
        delta=0.01,
        s_threshold=[0, 0, 0, 0, 0, 0]
    )

    # Display results
    print_results(utilities, errors, grids, cars, ranking)
    
    print("\n" + "=" * 60)
    print("Marginal Utility Functions:")
    print("=" * 60)
    for i, (u_vals, grid) in enumerate(zip(utilities, grids)):
        print(f"\nCriterion {i+1}:")
        for g, u in zip(grid, u_vals):
            print(f"  g = {g:8.2f} → u(g) = {u:.4f}")