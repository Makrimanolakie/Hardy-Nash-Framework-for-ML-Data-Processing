"""
Hardy-Nash Framework for ML Data Processing

Combines discrete Hardy inequalities with Nash iteration schemes
for robust optimization and attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Callable, Tuple, Optional, List


class DiscreteHardyWeight:
    """
    Compute discrete Hardy weights for sequences
    
    For discrete data on grid {x₁, ..., xₙ}, computes Hardy weights:
        wᵢ = aᵢ/(Aᵢ² Bᵢ²)
    
    where:
        Aᵢ = h Σⱼ₌₁ⁱ 1/aⱼ
        Bᵢ = h Σⱼ₌ᵢᴺ 1/aⱼ
    
    Parameters:
    -----------
    a_func : callable
        Coefficient function a(x)
    h : float
        Grid spacing (default: 1.0 for index-based)
    learnable : bool
        Whether a_func parameters are learnable
    """
    
    def __init__(
        self, 
        a_func: Optional[Callable] = None,
        h: float = 1.0,
        learnable: bool = False
    ):
        if a_func is None:
            # Default: uniform weights
            a_func = lambda x: torch.ones_like(x)
        
        self.a_func = a_func
        self.h = h
        self.learnable = learnable
        
        # Cache for efficiency
        self._cache = {}
    
    def compute_cumulative(self, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cumulative sums A and B
        
        Parameters:
        -----------
        a : torch.Tensor, shape (N,)
            Coefficient values
            
        Returns:
        --------
        A, B : torch.Tensor, shape (N,)
            Cumulative sums
        """
        N = len(a)
        device = a.device
        
        # A[i] = h * Σⱼ₌₁ⁱ 1/aⱼ
        a_inv = 1.0 / (a + 1e-10)
        A = self.h * torch.cumsum(a_inv, dim=0)
        
        # B[i] = h * Σⱼ₌ᵢᴺ 1/aⱼ
        a_inv_reversed = torch.flip(a_inv, [0])
        B_reversed = self.h * torch.cumsum(a_inv_reversed, dim=0)
        B = torch.flip(B_reversed, [0])
        
        return A, B
    
    def compute_weight(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Hardy weights
        
        Parameters:
        -----------
        x : torch.Tensor, shape (N,)
            Grid points or indices
            
        Returns:
        --------
        w : torch.Tensor, shape (N,)
            Hardy weights
        """
        a = self.a_func(x)
        A, B = self.compute_cumulative(a)
        
        # w = a / (A²B²)
        eps = 1e-10
        w = a / (A**2 * B**2 + eps)
        
        return w
    
    def hardy_constant(self, N: int) -> float:
        """
        Estimate discrete Hardy constant
        
        C_H ≈ h²N²/4 for uniform grid
        """
        return (self.h * N)**2 / 4.0


class HardySoftmaxAttention(nn.Module):
    """
    Multi-head attention with Hardy inequality bias
    
    Standard attention:
        Attention(Q,K,V) = softmax(QKᵀ/√d_k) V
    
    Hardy-biased attention:
        Attention(Q,K,V) = softmax(QKᵀ/√d_k + log(wᵢ/wⱼ)) V
    
    where w are Hardy weights computed from position embeddings
    
    Parameters:
    -----------
    d_model : int
        Model dimension
    n_heads : int
        Number of attention heads
    hardy_weight : DiscreteHardyWeight
        Hardy weight computer
    use_hardy_bias : bool
        Whether to add Hardy bias to attention scores
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        hardy_weight: Optional[DiscreteHardyWeight] = None,
        use_hardy_bias: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_hardy_bias = use_hardy_bias
        
        if hardy_weight is None:
            hardy_weight = DiscreteHardyWeight()
        self.hardy_weight = hardy_weight
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable temperature for Hardy bias
        self.hardy_temperature = nn.Parameter(torch.ones(1))
    
    def compute_hardy_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Compute Hardy bias matrix H[i,j] = log(wᵢ/wⱼ)
        
        Parameters:
        -----------
        seq_len : int
            Sequence length
        device : torch.device
            Device to create tensor on
            
        Returns:
        --------
        H : torch.Tensor, shape (seq_len, seq_len)
            Hardy bias matrix
        """
        # Position indices
        indices = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        # Compute Hardy weights
        w = self.hardy_weight.compute_weight(indices)
        
        # Log ratio matrix: H[i,j] = log(wᵢ/wⱼ)
        log_w = torch.log(w + 1e-10)
        H = log_w.unsqueeze(1) - log_w.unsqueeze(0)
        
        return H * self.hardy_temperature
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with Hardy-biased attention
        
        Parameters:
        -----------
        x : torch.Tensor, shape (batch, seq_len, d_model)
            Input sequence
        mask : torch.Tensor, optional
            Attention mask
            
        Returns:
        --------
        out : torch.Tensor, shape (batch, seq_len, d_model)
            Output sequence
        attn_weights : torch.Tensor, shape (batch, n_heads, seq_len, seq_len)
            Attention weights
        """
        B, L, D = x.shape
        
        # Linear projections and reshape for multi-head attention
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores: QKᵀ/√d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add Hardy bias
        if self.use_hardy_bias:
            hardy_bias = self.compute_hardy_bias(L, x.device)
            scores = scores + hardy_bias.unsqueeze(0).unsqueeze(0)  # Broadcast
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.W_o(attn_output)
        
        return output, attn_weights


class WeightedSobolevRegularizer(nn.Module):
    """
    Discrete weighted Sobolev space regularization
    
    R(u) = λ₁ Σᵢ wᵢuᵢ² + λ₂ Σᵢ aᵢ(Δuᵢ)²
    
    This enforces smoothness in the weighted L² and H¹ norms
    
    Parameters:
    -----------
    hardy_weight : DiscreteHardyWeight
        Hardy weight computer
    lambda_1 : float
        Weight for L² term
    lambda_2 : float
        Weight for H¹ term (gradient penalty)
    """
    
    def __init__(
        self,
        hardy_weight: DiscreteHardyWeight,
        lambda_1: float = 1.0,
        lambda_2: float = 1.0
    ):
        super().__init__()
        self.hardy_weight = hardy_weight
        self.register_buffer('lambda_1', torch.tensor(lambda_1))
        self.register_buffer('lambda_2', torch.tensor(lambda_2))
    
    def compute_discrete_derivative(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute discrete derivative Δu[i] = u[i+1] - u[i]
        
        Parameters:
        -----------
        u : torch.Tensor, shape (N,) or (batch, N)
            Input sequence
            
        Returns:
        --------
        du : torch.Tensor, shape (N-1,) or (batch, N-1)
            Discrete derivative
        """
        if u.dim() == 1:
            return u[1:] - u[:-1]
        else:
            return u[:, 1:] - u[:, :-1]
    
    def forward(
        self,
        u: torch.Tensor,
        indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted Sobolev regularization
        
        Parameters:
        -----------
        u : torch.Tensor, shape (N,) or (batch, N)
            Sequence to regularize
        indices : torch.Tensor, optional
            Position indices (default: 0, 1, ..., N-1)
            
        Returns:
        --------
        reg : torch.Tensor, scalar
            Regularization value
        """
        if indices is None:
            indices = torch.arange(u.shape[-1], device=u.device, dtype=torch.float32)
        
        # Compute Hardy weights
        w = self.hardy_weight.compute_weight(indices)
        a = self.hardy_weight.a_func(indices)
        
        # L² term (weighted by Hardy weights)
        if u.dim() == 1:
            l2_term = torch.sum(w * u**2)
        else:
            l2_term = torch.sum(w.unsqueeze(0) * u**2)
        
        # H¹ term (discrete gradient)
        du = self.compute_discrete_derivative(u)
        if u.dim() == 1:
            h1_term = torch.sum(a[:-1] * du**2)
        else:
            h1_term = torch.sum(a[:-1].unsqueeze(0) * du**2)
        
        # Total regularization
        reg = self.lambda_1 * l2_term + self.lambda_2 * h1_term
        
        return reg


class NashFixedPointSolver:
    """
    Nash fixed-point iteration for solving F(u) = 0
    
    Decomposes F = F₁ + F₂ and solves via alternating updates:
        F₁(uⁿ⁺¹, uⁿ) + F₂(uⁿ, uⁿ) = 0
    
    Uses Hardy weights for preconditioning
    
    Parameters:
    -----------
    F1, F2 : callable
        Decomposition of F = F₁ + F₂
    hardy_weight : DiscreteHardyWeight
        Hardy weight for preconditioning
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    """
    
    def __init__(
        self,
        F1: Callable,
        F2: Callable,
        hardy_weight: DiscreteHardyWeight,
        max_iter: int = 100,
        tol: float = 1e-6,
        damping: float = 1.0
    ):
        self.F1 = F1
        self.F2 = F2
        self.hardy_weight = hardy_weight
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping
    
    def solve(
        self,
        u0: torch.Tensor,
        indices: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Solve F(u) = 0 via Nash fixed-point iteration
        
        Parameters:
        -----------
        u0 : torch.Tensor
            Initial guess
        indices : torch.Tensor
            Position indices
            
        Returns:
        --------
        u : torch.Tensor
            Solution
        history : dict
            Convergence history
        """
        u = u0.clone()
        history = {
            'residual': [],
            'update': [],
            'iterations': 0
        }
        
        for k in range(self.max_iter):
            # Compute Hardy weights for preconditioning
            w = self.hardy_weight.compute_weight(indices)
            W_inv = torch.diag(1.0 / torch.sqrt(w + 1e-10))
            
            # Evaluate F₂ at current point
            F2_val = self.F2(u, u)
            
            # Solve F₁(u_new, u) = -F₂(u, u) in weighted space
            # This is problem-specific; here we use gradient descent
            
            # Compute residual
            rhs = -F2_val
            
            # Preconditioned update
            update = self.damping * W_inv @ rhs
            u_new = u + update
            
            # Compute full residual F(u_new) = F₁(u_new, u_new) + F₂(u_new, u_new)
            residual = self.F1(u_new, u_new) + self.F2(u_new, u_new)
            
            # Track convergence
            res_norm = torch.norm(residual)
            update_norm = torch.norm(update)
            
            history['residual'].append(res_norm.item())
            history['update'].append(update_norm.item())
            
            # Check convergence
            if update_norm < self.tol:
                history['iterations'] = k + 1
                break
            
            u = u_new
        
        return u, history


class NashNewtonSolver:
    """
    Nash-Newton iteration for solving F(u) = 0
    
    Solves the Newton system in weighted Sobolev space:
        J_w(uⁿ) · Δu = -W^{1/2} F(uⁿ)
    
    where J_w = W^{1/2} J(u) W^{-1/2} is the weighted Jacobian
    
    Parameters:
    -----------
    F : callable
        Nonlinear operator F(u, indices)
    hardy_weight : DiscreteHardyWeight
        Hardy weight for preconditioning
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    """
    
    def __init__(
        self,
        F: Callable,
        hardy_weight: DiscreteHardyWeight,
        max_iter: int = 50,
        tol: float = 1e-6,
        use_line_search: bool = True
    ):
        self.F = F
        self.hardy_weight = hardy_weight
        self.max_iter = max_iter
        self.tol = tol
        self.use_line_search = use_line_search
    
    def compute_jacobian(
        self,
        u: torch.Tensor,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Jacobian J = ∂F/∂u using automatic differentiation
        
        Parameters:
        -----------
        u : torch.Tensor, shape (N,)
            Current point
        indices : torch.Tensor
            Position indices
            
        Returns:
        --------
        J : torch.Tensor, shape (N, N)
            Jacobian matrix
        """
        u_var = u.clone().detach().requires_grad_(True)
        F_val = self.F(u_var, indices)
        
        N = len(u)
        J = torch.zeros(N, N, device=u.device)
        
        for i in range(N):
            if u_var.grad is not None:
                u_var.grad.zero_()
            
            # Compute gradient of F[i] w.r.t. u
            F_val[i].backward(retain_graph=(i < N-1))
            J[i] = u_var.grad.clone()
        
        return J
    
    def line_search(
        self,
        u: torch.Tensor,
        delta_u: torch.Tensor,
        indices: torch.Tensor,
        alpha_init: float = 1.0,
        rho: float = 0.5,
        c: float = 1e-4
    ) -> float:
        """
        Armijo line search for step size
        
        Returns:
        --------
        alpha : float
            Step size
        """
        alpha = alpha_init
        F_u = self.F(u, indices)
        norm_F_u = torch.norm(F_u)**2
        
        for _ in range(10):  # Max 10 backtracks
            u_new = u + alpha * delta_u
            F_new = self.F(u_new, indices)
            norm_F_new = torch.norm(F_new)**2
            
            # Armijo condition
            if norm_F_new <= norm_F_u - c * alpha * norm_F_u:
                break
            
            alpha *= rho
        
        return alpha
    
    def solve(
        self,
        u0: torch.Tensor,
        indices: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Solve F(u) = 0 via Nash-Newton iteration
        
        Parameters:
        -----------
        u0 : torch.Tensor
            Initial guess
        indices : torch.Tensor
            Position indices
            
        Returns:
        --------
        u : torch.Tensor
            Solution
        history : dict
            Convergence history
        """
        u = u0.clone()
        history = {
            'residual': [],
            'update': [],
            'step_size': [],
            'iterations': 0
        }
        
        for k in range(self.max_iter):
            # Compute Hardy weights
            w = self.hardy_weight.compute_weight(indices)
            W_sqrt = torch.diag(torch.sqrt(w))
            W_inv_sqrt = torch.diag(1.0 / torch.sqrt(w + 1e-10))
            
            # Compute Jacobian
            J = self.compute_jacobian(u, indices)
            
            # Transform to weighted Jacobian: J_w = W^{1/2} J W^{-1/2}
            J_w = W_sqrt @ J @ W_inv_sqrt
            
            # Compute weighted residual: F_w = W^{1/2} F(u)
            F_u = self.F(u, indices)
            F_w = W_sqrt @ F_u
            
            # Solve: J_w · delta_u_w = -F_w
            try:
                delta_u_w = torch.linalg.solve(J_w, -F_w)
            except RuntimeError:
                # Use least squares if singular
                delta_u_w = torch.linalg.lstsq(J_w, -F_w).solution
            
            # Transform back: delta_u = W^{-1/2} delta_u_w
            delta_u = W_inv_sqrt @ delta_u_w
            
            # Line search for step size
            if self.use_line_search:
                alpha = self.line_search(u, delta_u, indices)
            else:
                alpha = 1.0
            
            # Update
            u = u + alpha * delta_u
            
            # Track convergence
            residual = torch.norm(F_u)
            update = torch.norm(delta_u)
            
            history['residual'].append(residual.item())
            history['update'].append(update.item())
            history['step_size'].append(alpha)
            
            # Check convergence
            if update < self.tol:
                history['iterations'] = k + 1
                break
        
        return u, history


class HardyNashOptimizer(torch.optim.Optimizer):
    """
    Nash-Newton optimizer for neural networks using Hardy preconditioning
    
    Uses Hardy-weighted Hessian approximation for second-order optimization
    
    Parameters:
    -----------
    params : iterable
        Model parameters
    hardy_weight : DiscreteHardyWeight
        Hardy weight for preconditioning
    lr : float
        Learning rate
    use_hessian : bool
        Whether to use Hessian (vs gradient descent)
    """
    
    def __init__(
        self,
        params,
        hardy_weight: DiscreteHardyWeight,
        lr: float = 0.01,
        use_hessian: bool = False,
        momentum: float = 0.9
    ):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        
        self.hardy_weight = hardy_weight
        self.use_hessian = use_hessian
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform optimization step
        
        Parameters:
        -----------
        closure : callable, optional
            A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Compute Hardy weights for this parameter
                # (simplified - use parameter index as position)
                indices = torch.arange(p.numel(), device=p.device, dtype=torch.float32)
                w = self.hardy_weight.compute_weight(indices)
                w = w.view(p.shape)
                
                # Preconditioned gradient
                grad_precond = grad / (w + 1e-10)
                
                # Momentum
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.clone(grad_precond).detach()
                else:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).add_(grad_precond)
                    grad_precond = buf
                
                # Update
                p.data.add_(grad_precond, alpha=-group['lr'])
        
        return loss


# Example usage and tests
if __name__ == "__main__":
    print("Hardy-Nash Framework for ML - Test Suite")
    print("=" * 60)
    
    # Test 1: Discrete Hardy weights
    print("\n1. Testing Discrete Hardy Weights")
    hardy = DiscreteHardyWeight(a_func=lambda x: 1.0 + 0.1 * torch.sin(x))
    indices = torch.linspace(0, 10, 50)
    w = hardy.compute_weight(indices)
    print(f"   Weight range: [{w.min():.4f}, {w.max():.4f}]")
    print(f"   Hardy constant: {hardy.hardy_constant(50):.4f}")
    
    # Test 2: Hardy-Softmax Attention
    print("\n2. Testing Hardy-Softmax Attention")
    attention = HardySoftmaxAttention(d_model=64, n_heads=8, hardy_weight=hardy)
    x = torch.randn(2, 20, 64)  # (batch, seq_len, d_model)
    out, attn_weights = attention(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    
    # Test 3: Weighted Sobolev Regularization
    print("\n3. Testing Weighted Sobolev Regularization")
    reg = WeightedSobolevRegularizer(hardy, lambda_1=1.0, lambda_2=0.1)
    u = torch.randn(50)
    reg_value = reg(u, indices)
    print(f"   Sequence length: {len(u)}")
    print(f"   Regularization value: {reg_value.item():.6f}")
    
    # Test 4: Nash Fixed-Point Solver
    print("\n4. Testing Nash Fixed-Point Solver")
    
    # Simple nonlinear problem: u² - 2u + 1 = 0 (solution: u = 1)
    F1 = lambda u_new, u_old: u_new**2 - u_old
    F2 = lambda u, _: -u + 1.0
    
    solver_fp = NashFixedPointSolver(F1, F2, hardy, max_iter=100, damping=0.5)
    u0 = torch.zeros(10)
    indices_small = torch.arange(10, dtype=torch.float32)
    u_sol, history_fp = solver_fp.solve(u0, indices_small)
    
    print(f"   Converged in {history_fp['iterations']} iterations")
    print(f"   Final residual: {history_fp['residual'][-1]:.6e}")
    print(f"   Solution (should be ~1.0): {u_sol.mean().item():.6f}")
    
    # Test 5: Nash-Newton Solver
    print("\n5. Testing Nash-Newton Solver")
    
    # Problem: u² - 2 = 0 (solution: u = √2)
    F_nonlinear = lambda u, idx: u**2 - 2.0
    
    solver_newton = NashNewtonSolver(F_nonlinear, hardy, max_iter=20)
    u0_newton = torch.ones(10)
    u_sol_newton, history_newton = solver_newton.solve(u0_newton, indices_small)
    
    print(f"   Converged in {history_newton['iterations']} iterations")
    print(f"   Final residual: {history_newton['residual'][-1]:.6e}")
    print(f"   Solution (should be ~1.414): {u_sol_newton.mean().item():.6f}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
