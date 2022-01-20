from properror import MatrixEquation, compute_uncertainties, compute_jacobian
import numpy as np
import time

def test_matrix_size():
    import time
    t_solve = []
    for N in range(2, 12):
        print(f'size(A) = {N}x{N}')
        d = np.arange(N)
        A = np.identity(N)
        cov_A = np.zeros(shape=[N,]*4)
        cov_d = np.identity(N)
        print(d.shape, A.shape, cov_A.shape, cov_d.shape)
        meq = MatrixEquation(A, d, cov_A, cov_d, N=N)
        t0 = time.time()
        x, cov_x = meq.solve()
        t1 = time.time()
        t_solve.append(t1-t0)
        print(f'Solution time: {t1-t0}s')
    print(t_solve)

def test_prop():
    from sympy import symbols, sqrt, diff, evalf
    from uncertainties import ufloat, unumpy

    fvars = symbols('a b c d')
    func = fvars[0] + fvars[1] + sqrt(fvars[2])
    func2 = sqrt(fvars[1] + 2*fvars[2])
    fvals = {fvars[0]: 100, fvars[1]: 200, fvars[2]: 250, fvars[3]: 500}

    func_list = [func, func2]
    cvm = np.identity(4)
    compute_jacobian(func_list, fvars, fvals)
    compute_uncertainties(func_list, fvars, fvals, cvm)

if __name__ == "__main__":
    test_matrix_size()
    test_prop()