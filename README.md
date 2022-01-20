## properror
A simple code for error propagation during matrix inversion in Python.

### Propagating errors in matrix inversion

Consider the seemingly simple matrix equation:

<img src="https://render.githubusercontent.com/render/math?math={\bf x}={\bf A}^{-1}{\bf d}">

where **A** is an *(NxN)* matrix, with known uncertainties of each element, and **d** is an *(nx1)* vector with known uncertainties. What are the uncertainties on the entries of **x**? 

The solution is surprisingly complex [[1]](#1): the errors (i.e. covariances) between entries _x<sub>i</sub>_ and _x<sub>j</sub>_ is given by:

<img src="https://render.githubusercontent.com/render/math?math={\rm{cov}}(x_{i},x_{j})=d_{\alpha}d_{\beta}{\rm{cov}}(A_{i\alpha}^{-1},A_{j\beta}^{-1})%2B A_{ik}^{-1}A_{jl}^{-1}{\rm{cov}}(d_{k},d_{l})">

This equation is written in Einstein summation notation, where repeated indices are summed over. To solve this, we first need to compute the covariances of all the entries in the inverse matrix **A<sup>-1</sup>**, which is given by 

<img src="https://render.githubusercontent.com/render/math?math={\rm{cov}}(A_{\alpha\beta}^{-1},A_{ab}^{-1})=A_{\alpha i}^{-1}A_{j\beta}^{-1}A_{ak}^{-1}A_{lb}^{-1}{\rm{cov}}(A_{ij,}A_{kl})">

This Python package computes the solution to **x=A<sup>-1</sup>y** and corresponding covariances. 

### Usage example

`properror` is quite straightforward to use:

```python
import numpy as np
from properror import MatrixEquation

# Create data, A matrix, and covariances
# cov_A
N = 4    # 4x4 matrix
d = np.arange(N)                  # Example input data
A = np.identity(N)                # Matrix A 
cov_A = np.zeros(shape=[N,N,N,N]) # Covariance between A_ij and A_kl, shape (N, N, N, N)
cov_d = np.identity(N)            # Covariance between data, d_i and d_j, shape (N, N)

# Instantiate the matrix equation and solve
meq = MatrixEquation(A, d, cov_A, cov_d, N=N)
x, cov_x = meq.solve()
```
Note that the number of entries in the covariance matrix of **A** scales with *N<sup>4*, so *N<=16* is recommended unless you need a nap.

### Propagating using sympy functions

After your matrix inversion you may need to apply some transforms (e.g. cosin, sqrt, 1/x, etc) to continue your merry uncertainty quest. There's a `compute_uncertainties` method to do so. 

This function applies the matrix expression for error propagation [[2]](#2)

<img src="https://render.githubusercontent.com/render/math?math=\sigma_{f}^{2}=\mathbf{g}^{T}\mathbf{V}\mathbf{g}">

where <img src="https://render.githubusercontent.com/render/math?math=\sigma_{f}^{2}"> is the variance for function f of a set of parameters <img src="https://render.githubusercontent.com/render/math?math=\beta">, with a variance-covariance matrix **V**. The vector **g** is the Jacobian vector, with the ith element given by <img src="https://render.githubusercontent.com/render/math?math=\partial f/\partial\beta_{i}">.

Fortunately, `compute_uncertainties` will handle the partial differentiation, by making use of the sympy symbolic math package:

```python
from sympy import symbols, sqrt, diff, evalf

fvars = symbols('a b c d')

# Create some functions
# f = a + b + sqrt(c)
func = fvars[0] + fvars[1] + sqrt(fvars[2])
# f2 = sqrt(a + 2b)
func2 = sqrt(fvars[1] + 2*fvars[2])
func_list = [func, func2]

# Expected values dictionary
fvals = {
        fvars[0]: 100,  # a = 100
        fvars[1]: 200,  # b = 200
        fvars[2]: 250,  # c = 250 
        fvars[3]: 500   # d = 500
    }


# Covariance matrix (can user output from meq.solve())
cvm = np.identity(4)

u = compute_uncertainties(func_list, fvars, fvals, cvm)
```

## References 
<a id="1">[1]</a> M. Lefebvre, R. K. Keeler, R. Sobie, and J. White. Propagation of errors
for matrix inversion. Nuclear Instruments and Methods in Physics Research
A, 451(2):520–528, September 2000.

<a id="2">[2]</a> Joel Tellinghuisen. Statistical Error Propagation. Journal of Physical Chemistry
A, 105(15):3917–3921, April 2001.
