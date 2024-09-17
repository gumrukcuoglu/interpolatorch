#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:03:06 2024

Some interpolation functions/classes that are compatible with autograd in torch

@author: emirg
"""


import torch


class InterpolateLinear:
    """
    Given knots (z_knots, f_knots) generate linear interpolation 
    function f(z). 
    
    z_knots: torch tensor (shape N) [should be in ascending order]
    f_knots: torch tensor (shape N)
    extrapolate: bool, if True extrapolates according to slopes at boundary
                          False assigns nan to out-of-bound values.
    
    Only accepts torch tensors for the moment.
    """
    
    def __init__(self, z_knots, f_knots, extrapolate=False):
        self.z_knots = z_knots
        self.f_knots = f_knots
        self.extrapolate = extrapolate
        
        # Step sizes
        h = torch.diff(self.z_knots) 
        df = torch.diff(self.f_knots)
        
        # Linear function a z +b based on left neighbour:
        self.a= df/h
        self.b= f_knots[:-1]
        
    def __call__(self, z_list):
        
        # Flatten z_list to 1D for easier processing
        z_flat = z_list.flatten()
        
        # Compute the indices for the left neighbors
        L_idx = torch.clamp(torch.searchsorted(self.z_knots, z_flat) - 1, 0, len(self.z_knots) - 2)
        
        # Left neighbour of each point
        z_L = self.z_knots[L_idx]
        
        dz = z_flat - z_L
        f_flat = self.a[L_idx] * dz + self.b[L_idx]

        if not self.extrapolate:
            # Set out-of-bound values to NaN
            f_flat = torch.where((z_flat < self.z_knots[0]) | (z_flat > self.z_knots[-1]), torch.tensor(float('nan'), device=z_flat.device, dtype=z_flat.dtype), f_flat)
        
        # Reshape to match the input list 
        f_list = f_flat.view_as(z_list)
        
        return f_list
    
    
class CubicSplines:
    """
    Given knots (z_knots, f_knots) generate cubic spline interpolation 
    function f(z). 
    Uses natural boundary conditions.
    
    z_knots: torch tensor (shape N) [should be in ascending order]
    f_knots: torch tensor (shape N)
    extrapolate: bool, if True extrapolates using the function at the boundaries.
                          False assigns nan to out-of-bound values.
    
    Only accepts torch tensors for the moment.
    """
    
    def __init__(self, z_knots, f_knots, extrapolate=True):
        self.z_knots = z_knots
        self.f_knots = f_knots
        self.extrapolate = extrapolate
        
        # Number of intervals
        n = len(z_knots) - 1
        
        # Step sizes
        h = torch.diff(z_knots) 
        
        # Solve for spline coefficients using the tridiagonal matrix algorithm (TDMA)
        
        
        # alpha_i = 3 (f_{i+2} - f_{i+1}) / h_{i+1} - 3 (f_{i+1} - f_{i}) / h_{i}
        f_by_h_diff = (f_knots[1:] - f_knots[:-1]) / h
        alpha = 3*torch.diff(f_by_h_diff)
        
        
        # Create a dense tridiagonal matrix
        A = torch.zeros((n-1, n-1), device=h.device, dtype=h.dtype)
        
        # Fill the three diagonals
        A[range(n-1), range(n-1)] = 2 * (h[1:] + h[:-1])
        A[range(n-2), range(1, n-1)] = h[1:-1]
        A[range(1, n-1), range(n-2)] = h[1:-1]
        
        c_mid = torch.linalg.solve(A, alpha)
        c_edges = torch.zeros(1, device=h.device, dtype=h.dtype)
        
        # Compute the other coefficients
        self.c = torch.hstack([c_edges, c_mid, c_edges])
        self.b = f_by_h_diff - h * (2 * self.c[:-1] + self.c[1:]) / 3
        self.d = torch.diff(self.c) / (3 * h)
        self.a = f_knots[:-1]
    
    def __call__(self, z_list):
        # Flatten z_list to 1D for easier processing
        z_flat = z_list.flatten()
        
        # Compute the indices for the left neighbors
        L_idx = torch.clamp(torch.searchsorted(self.z_knots, z_flat) - 1, 0, len(self.z_knots) - 2)
        
        z_L = self.z_knots[L_idx]
        
        dz = z_flat - z_L
        f_flat = self.a[L_idx] + self.b[L_idx] * dz + self.c[L_idx] * dz**2 + self.d[L_idx] * dz**3

        if not self.extrapolate:
            # Set out-of-bound values to NaN
            f_flat = torch.where((z_flat < self.z_knots[0]) | (z_flat > self.z_knots[-1]), torch.tensor(float('nan'), device=z_flat.device, dtype=z_flat.dtype), f_flat)

        f_list = f_flat.view_as(z_list)
        return f_list
