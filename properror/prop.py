import numpy as np
from sympy import symbols, sqrt, diff, evalf
from uncertainties import ufloat, unumpy

def compute_jacobian(func_list, func_vars, values):
    """ Compute jacobian for sympy func and evaluate given values 
    
    Arguments:
        func_list (list of sympy functions)
    
    Returns:
        jacv (np.array): Evaluated Jacobian vector
    """
    jac = [[diff(f, v) for v in func_vars] for f in func_list]
    jacv = np.matrix([[float(z.evalf(subs=values)) for z in j] for j in jac]).T
    return jacv

def compute_uncertainties(func_list, func_vars, values_dict, covariances):
    """ Given a function, with variables func_vars and covariance cvm, compute uncertainties 
    
    Use the propagation of errors var = gT V g
    g = jacobian vector formed from func (analytically, then numerically evaluated)
    V = covariance matrix cvm
    
    Arguments:
        func (sympy function): sympy function that relates the variables
        func_vars (list): list of sympy variables 
        values_dict (dict): dictionary of N values
        covariances (np.array): NxN variance-covariance matrix
    
    Returns:
        val (ufloat): Computed value and uncertainty in ufloat
    """
    v0 = [func.evalf(subs=values_dict) for func in func_list]
    jacv = compute_jacobian(func_list, func_vars, values_dict)
    var_func = jacv.T * np.matrix(covariances) * jacv
    std_func = np.sqrt(var_func)
    return unumpy.uarray(v0, std_func)