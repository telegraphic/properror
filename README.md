## properror
Basic error propagation for matrix inversion in Python.

### Propagating errors in matrix inversion

Consider the seemingly simple matrix equation:

<img src="https://render.githubusercontent.com/render/math?math={\bf x}={\bf A}^{-1}{\bf y}">

where **A** is an *(NxN)* matrix, with known uncertainties of each element, and **y** is an *(nx1)* vector with known uncertainties. What are the uncertainties on the entries of **x**? 

The solution is surprisingly complex [[1]](#1): the errors (i.e. covariances) between entries _x<sub>i</sub>_ and _x<sub>j</sub>_ is given by:

<img src="https://render.githubusercontent.com/render/math?math={\rm{cov}}(x_{i},x_{j})=y_{\alpha}y_{\beta}{\rm{cov}}(A_{i\alpha}^{-1},A_{j\beta}^{-1})%2B A_{ik}^{-1}A_{jl}^{-1}{\rm{cov}}(y_{k},y_{l})">

This equation is written in Einstein summation notation, where repeated indices are summed over. To solve this, we first need to compute the covariances of all the entries in the inverse matrix **A<sup>-1</sup>**, which is given by 

<img src="https://render.githubusercontent.com/render/math?math={\rm{cov}}(A_{\alpha\beta}^{-1},A_{ab}^{-1})=A_{\alpha i}^{-1}A_{j\beta}^{-1}A_{ak}^{-1}A_{lb}^{-1}{\rm{cov}}(A_{ij,}A_{kl})">

This Python package computes the solution to **x=A<sup>-1</sup>y** and corresponding covariances. Here's a usage example:

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
Note that the number of entries in the covariance matrix of **A** scales with *N<sup>4*, so N<=8 is recommended as the code is not optimized for speed.


## References 
<a id="1">[1]</a> M. Lefebvre, R. K. Keeler, R. Sobie, and J. White. Propagation of errors
for matrix inversion. Nuclear Instruments and Methods in Physics Research
A, 451(2):520–528, September 2000.
