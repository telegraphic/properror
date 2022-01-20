import numpy as np

class MatrixEquation(object):
    def __init__(self, A, d, covA, covd, N=4):
        """ Solve matrix equation x = AI d with uncertainties
        
        Computes uncertainties on x = AI d, following Lefebvre et al (2000) 
        TODO: Investigate use of np.einsum
        
        Args:
            d: measured data vector with shape (Nx1)
            A: matrix with shape (NxN)
            covA: is covariance matrix with shape (NxNxNxN)
            covd: is covariance matrix with shape (NxN)
        
        Returns:
            x: vector with shape (Nx1)
        """
        A = np.matrix(A)
        self.A = A
        self.d = d
        self.AI = A.I
        self.N = N
        self.cov_A = covA
        self.cov_d = covd
        self.x = None
        self.cov_AI = None
        self.cov_x = None
        self.cov_x_p1 = None
        self.cov_x_p2 = None
    
    def solve(self):
        """ Solve matrix equation 
        
        Returns:
            x, cx: (Nx1) vector x and covariance matrix for x (NxN)
        """
        self.x = self.AI * np.matrix(self.d).T
        self._compute_cov_AI()
        self._compute_cov_x()
        return self.x, self.cov_x
    
    def _compute_cov_AI(self):
        """ Compute covariance of AI 
        
        Following Lefebvre (2000), eqn 7
        In Einstein summation notation:
        cov(AIab, AIcd) = AIai AIjb AIak AIlb cov(Aij, Akl)
        """
        AI = self.AI
        self.cov_AI = np.einsum('Ai,jB,ak,lb,ijkl', AI, AI, AI, AI, self.cov_A)
        return self.cov_AI
    
    def _compute_cov_x(self):
        """ Compute covariance of x (solution vector)
        
        Following Lefebvre (2000), eqn 11
        In Einstein summation notation:
        cov(xi, xj) = ya yb cov(AIia, AIjb) + AIij AIjl cov(yk yl)
        """
        if self.cov_AI is None:  
            self._compute_cov_AI()
        self.cov_x_p1 = np.einsum('A,B,iAjB',self.d, self.d, self.cov_AI)
        self.cov_x_p2 = np.einsum('ik,jl,kl', self.AI, self.AI, self.cov_d)
        self.cov_x = self.cov_x_p1 + self.cov_x_p2
        return self.cov_x
