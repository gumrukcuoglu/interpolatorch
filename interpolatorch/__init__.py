#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
interpolatorch v0.2

Some interpolation functions/classes that are compatible with autograd in torch

@author: emirg

History: 
    v0.1    11/7/2024 - Initial version, single function interpolation
    v0.2    18/9/2024 - Implemented parallel handling of independent interpolations.
                        Old version still available with "_legacy" suffix.
            20/9/2024 - Fixed a forgotten contiguous conversion for already batched parameters
"""




import torch


class InterpolateLinear:
    
    """
    Given N_b independent knots (z_knots, f_knots) generate N_b linear 
    interpolations function f(z). 
    
    z_knots: torch tensor (shape N_b x N_t) [second axis in ascending order]
    f_knots: torch tensor (shape N_b x N_t)
    
    extrapolate: bool, if True extrapolates according to selection ext.
                          False assigns nan to out-of-bound values.
    ext: int, if extrapolate==True, one of following options is considered:
                ext=0 : extend the line at the closest boundary
                ext=1 : constant, set to closest boundary value 
                ext=2 : constant, set to boundary values in ext_value. Ignored
                        if ext_value==None.

    ext_value: list of length 2, [left_boundary_value, right_boundary_value].
                Sets the same extrapolation value for all interpolated functions.
                Ignored unless extrapolate==True and ext==2.
                
    All inputs need to be torch tensors.
    """
    
    def __init__(self, z_knots, f_knots, extrapolate=False, ext=0, ext_value=None):
        # If z_knots non-contiguous, make it so
        self.z_knots = z_knots if z_knots.is_contiguous() else z_knots.contiguous()
        # self.z_knots = z_knots
        self.f_knots = f_knots
        self.extrapolate = extrapolate # Should I extrapolate?
        self.ext = ext # extrapolation style
        self.ext_value = ext_value # only used if ext==2.
        self.N_b, *_, self.N_t = z_knots.shape  # Keep number of independent 
                                                # data and number of z_knots

        # Step sizes
        h = torch.diff(self.z_knots, dim=-1) 
        df = torch.diff(self.f_knots, dim=-1)
        
        # Linear function az+b based on left neighbour:
        self.a= df/h
        self.b= self.f_knots[...,:-1]
        self.L_edges = self.z_knots[:,:1]
        self.R_edges = self.z_knots[:,-1:]
        
        if (ext not in [0,1,2]) and extrapolate:
            raise Exception("Extrapolation requested, but ext should be either 0, 1 or 2.")
            
        # Recording boundary values
        if (ext_value==None) or (ext==1): 
            # Values at left/right boundary
            self.f_L = self.b[:,:1]
            self.f_R = self.a[:,-1:]*(self.z_knots[:,-1:]-self.z_knots[:,-2:-1]) + self.b[:,-1:]
        elif ext==2:
            try:
                self.f_L, self.f_R = ext_value
            except:
                raise Exception("ext_value should be a list/tuple/tensor of size 2. ext=2 option sets the same boundary values for all data points.")
        
        # nan tensor that might be needed later repeatedly
        self._nan = torch.tensor(float("nan"), device=f_knots.device, dtype=f_knots.dtype)
        


    def __call__(self, z_list):
        
        # If separate z lists not provided, expand:
        if len(z_list) == self.N_b:
            z_expand = z_list if z_list.is_contiguous() else z_list.contiguous()
        else:
            z_expand = z_list if z_list.is_contiguous() else z_list.contiguous()
            z_expand = z_expand.expand(torch.Size([self.N_b]) + z_list.shape)
        
        
        # Flatten z_list to 1D for easier processing
        z_flat = z_expand.flatten(start_dim=1)
        z_flat = z_flat if z_flat.is_contiguous() else z_flat.contiguous()

        # Compute the indices for the left neighbors
        L_idx = torch.clamp(torch.searchsorted(self.z_knots, z_flat) - 1, 0, self.N_t - 2)
        
        # Left neighbour of each point
        z_L = torch.gather(self.z_knots, dim=1, index=L_idx)
        
        dz = z_flat - z_L
        f_flat = torch.gather(self.a, 1, L_idx) * dz + torch.gather(self.b, 1, L_idx)



        if not self.extrapolate or self.ext >0: # Do we need to mask the out-of-bound region?
            L_mask = z_flat < self.L_edges # left of left boundary
            R_mask = z_flat > self.R_edges # right of right boundary

        
        if not self.extrapolate: # Set out-of-bound values to NaN
            # f_flat = torch.where(L_mask | R_mask, 
            #                      torch.tensor(float('nan'), device=z_flat.device, dtype=z_flat.dtype), 
            #                      f_flat)
            
            f_flat = torch.where(L_mask | R_mask, self._nan, f_flat)
        elif self.ext>0: # Set out-of-bound values to the pre-defined constant
                    
            f_flat = torch.where(L_mask, self.f_L, f_flat)
            f_flat = torch.where(R_mask, self.f_R, f_flat)
            
        # Reshape to match the input list
        f_list = f_flat.view_as(z_expand)
        
        return f_list

class CubicSplines:
    
    """
    Given N_b independent knots (z_knots, f_knots) generate N_b cubic spline  
    interpolations function f(z). 
    
    For now, uses natural boundary conditions only.
    
    z_knots: torch tensor (shape N_b x N_t) [second axis in ascending order]
    f_knots: torch tensor (shape N_b x N_t)
    
    extrapolate: bool, if True extrapolates according to selection ext.
                          False assigns nan to out-of-bound values.
    ext: int, if extrapolate==True, one of following options is considered:
                ext=0 : extend the line at the closest boundary
                ext=1 : constant, set to closest boundary value 
                ext=2 : constant, set to boundary values in ext_value. Ignored
                        if ext_value==None.
        Note that ext=1 and 2 values introduce discontinuity at the boundary.
                

    ext_value: list of length 2, [left_boundary_value, right_boundary_value].
                Sets the same extrapolation value for all interpolated functions.
                Ignored unless extrapolate==True and ext==2.
                
    All inputs need to be torch tensors.
    """
    
    def __init__(self, z_knots, f_knots, extrapolate=False, ext=0, ext_value=None):
        self.z_knots = z_knots
        self.f_knots = f_knots
        self.extrapolate = extrapolate # Should I extrapolate?
        self.ext = ext # extrapolation style
        self.ext_value = ext_value # only used if ext==2.
        self.N_b, *_, self.N_t = z_knots.shape  # Keep number of independent 
                                                # data and number of z_knots

        n = self.N_t-1

        # Step sizes
        h = torch.diff(self.z_knots, dim=-1) 
        df = torch.diff(self.f_knots, dim=-1)

        # Solve for spline coefficients using the tridiagonal matrix algorithm (TDMA)       
        alpha = 3*torch.diff(df/h, dim=-1)
        
        # Create a dense tridiagonal matrix
        A = torch.zeros((self.N_b, n-1, n-1), device=h.device, dtype=h.dtype)


        # Fill the three diagonals
        A[:, range(n-1), range(n-1)] = 2 * (h[:,1:] + h[:,:-1])
        A[:, range(n-2), range(1, n-1)] = h[:,1:-1]
        A[:, range(1, n-1), range(n-2)] = h[:,1:-1]
        
        c_mid = torch.linalg.solve(A, alpha)
        c_edges = torch.zeros((self.N_b,1), device=h.device, dtype=h.dtype)
        
        # Compute the coefficients for y = a + b dz + c dz^2 +d dz^3
        self.c = torch.hstack([c_edges, c_mid, c_edges])
        self.b = df/h - h * (2 * self.c[:,:-1] + self.c[:,1:]) / 3
        self.d = torch.diff(self.c, dim=-1) / (3 * h)
        self.a = f_knots[:,:-1]

        self.L_edges = z_knots[:,:1]
        self.R_edges = z_knots[:,-1:]
        
        if (ext not in [0,1,2]) and extrapolate:
            raise Exception("Extrapolation requested, but ext should be either 0, 1 or 2.")
            
        # Recording boundary values
        if (ext_value==None) or (ext==1): 
            # Values at left/right boundary
            self.f_L = self.a[:,:1]
            dz_R = z_knots[:,-1:]-z_knots[:,-2:-1]
            self.f_R = self.a[:,-1:] + self.b[:,-1:]*dz_R + self.c[:,-2:-1]*dz_R**2 + self.d[:,-1:]*dz_R**3 
        elif ext==2:
            try:
                self.f_L, self.f_R = ext_value
            except:
                raise Exception("ext_value should be a list/tuple/tensor of size 2. ext=2 option sets the same boundary values for all data points.")
        
    
    def __call__(self, z_list):
        
        # If separate z lists not provided, expand:
        if len(z_list) == self.N_b:
            z_expand = z_list.contiguous()
        else:
            z_expand = z_list.expand(torch.Size([self.N_b]) + z_list.shape).contiguous()
        
        
        # Flatten z_list to 1D for easier processing
        z_flat = z_expand.flatten(start_dim=1)
        
        # Compute the indices for the left neighbors
        L_idx = torch.clamp(torch.searchsorted(self.z_knots, z_flat) - 1, 0, self.N_t - 2)
        
        # Left neighbour of each point
        z_L = torch.gather(self.z_knots, dim=1, index=L_idx)
        
        dz = z_flat - z_L
        
        f_flat = torch.gather(self.a, 1, L_idx) + torch.gather(self.b, 1, L_idx) * dz + torch.gather(self.c, 1, L_idx) * dz**2 + torch.gather(self.d, 1, L_idx) * dz**3
        
        if not self.extrapolate or self.ext >0: # Do we need to mask the out-of-bound region?
            L_mask = z_flat < self.L_edges # left of left boundary
            R_mask = z_flat > self.R_edges # right of right boundary

        
        if not self.extrapolate: # Set out-of-bound values to NaN
            f_flat = torch.where(L_mask | R_mask, 
                                 torch.tensor(float('nan'), device=z_flat.device, dtype=z_flat.dtype), 
                                 f_flat)
        elif self.ext>0: # Set out-of-bound values to the pre-defined constant
                    
            f_flat = torch.where(L_mask, self.f_L, f_flat)
            f_flat = torch.where(R_mask, self.f_R, f_flat)
            
        # Reshape to match the input list
        f_list = f_flat.view_as(z_expand)
        
        return f_list


class PCHIP:
    
    """
    
    Given N_b independent sets of ordered knots (z_knots, f_knots) generate N_b 
    monotonic interpolations f(z). 
      
    z_knots: torch tensor (shape N_b x N_t) [second axis in ascending order]
    f_knots: torch tensor (shape N_b x N_t) [monotonic values assumed]

    Extrapolation not implemented. Any f(z) for z \notin [min(z_knots), max(z_knots] will 
    evaluate to nan.
    
    All inputs need to be torch tensors.
    See Fritsch/Carlson 1980 and
        Fritsch/Butland 1984
    """
    
    def __init__(self, z_knots, f_knots, extrapolate=False, ext=0, ext_value=None):
        self.z_knots = z_knots
        self.f_knots = f_knots
        self.N_b, *_, self.N_t = z_knots.shape  # Keep number of independent 
                                                # data and number of z_knots

        # Step sizes
        h = torch.diff(self.z_knots, dim=-1) 
        df = torch.diff(self.f_knots, dim=-1)

        # 2-point right derivative:
        Delta = df/h
        # Inverses 1/Delta_i and 1/Delta_{i-1}
        atleD_i = 1./Delta[...,1:]
        atleD_im = 1./Delta[...,:-1]
        # Weight for the derivative estimate
        gamma = (1+1/(h[...,:-1]/h[...,1:]+1))/3

        # Determine points where Delta vanishes
        zero_mask = Delta ==0
        # And include i+1
        zero_mask[..., 1:] |= zero_mask[..., :-1].clone()

        # Estimate the first derivative [a la Fritsch/Butland G(S1,S2,h1,h2)]
        d_i = torch.zeros_like(f_knots)
        d_i[..., 1:-1] = (1./(gamma*atleD_im + (1-gamma)*atleD_i))

        # Set the boundary as the 2-point finite difference (one-sided)
        d_i[..., 0] = Delta[...,0]
        d_i[..., -1] = Delta[...,-1]
        # Set the derivatives to zero for repeating points.
        d_i[...,:-1][zero_mask]= 0.

        # Compute the coefficients for y = a + b dz + c dz^2 +d dz^3
        self.a = f_knots[...,:-1]
        self.b = d_i[...,:-1]
        self.c = (3*Delta - 2*d_i[...,:-1] - d_i[...,1:])/h
        self.d = (d_i[...,:-1] + d_i[...,1:] - 2*Delta)/h**2

        self.L_edges = z_knots[:,:1]
        self.R_edges = z_knots[:,-1:]

        # Values at left/right boundary
        self.f_L = self.a[...,:1]
        dz_R = z_knots[...,-1:]-z_knots[...,-2:-1]
        self.f_R = self.a[...,-1:] + self.b[...,-1:]*dz_R + self.c[...,-1:]*dz_R**2 + self.d[...,-1:]*dz_R**3 
    
    def __call__(self, z_list):
        
        # If separate z lists not provided, expand:
        if len(z_list) == self.N_b:
            z_expand = z_list.contiguous()
        else:
            z_expand = z_list.expand(torch.Size([self.N_b]) + z_list.shape).contiguous()
        
        # Flatten z_list to 1D for easier processing
        z_flat = z_expand.flatten(start_dim=1)
        
        # Compute the indices for the left neighbors
        L_idx = torch.clamp(torch.searchsorted(self.z_knots, z_flat) - 1, 0, self.N_t - 2)

        # Left neighbour of each point
        z_L = torch.gather(self.z_knots, dim=1, index=L_idx)
        
        dz = z_flat - z_L
     
        f_flat = torch.gather(self.a, 1, L_idx) + torch.gather(self.b, 1, L_idx) * dz + torch.gather(self.c, 1, L_idx) * dz**2 + torch.gather(self.d, 1, L_idx) * dz**3
        
        L_mask = z_flat < self.L_edges # left of left boundary
        R_mask = z_flat > self.R_edges # right of right boundary
      
        f_flat = torch.where(L_mask | R_mask, 
                             torch.tensor(float('nan'), device=z_flat.device, dtype=z_flat.dtype), 
                             f_flat)
        
        # Reshape to match the input list
        f_list = f_flat.view_as(z_expand)
        
        return f_list


### LEGACY CODES from v0.1

class InterpolateLinear_legacy:
    """
    Given knots (z_knots, f_knots) generate linear interpolation 
    function f(z). 
    
    z_knots: torch tensor (shape N) [should be in ascending order]
    f_knots: torch tensor (shape N)
    extrapolate: bool, if True extrapolates according to slopes at boundary
                          False assigns nan to out-of-bound values.
    
    Legacy function. Use InterpolateLinear for batch interpolation.
    
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
    
    
class CubicSplines_legacy:
    """
    Given knots (z_knots, f_knots) generate cubic spline interpolation 
    function f(z). 
    Uses natural boundary conditions.
    
    z_knots: torch tensor (shape N) [should be in ascending order]
    f_knots: torch tensor (shape N)
    extrapolate: bool, if True extrapolates using the function at the boundaries.
                          False assigns nan to out-of-bound values.
    
    Legacy function. Use CubicSplines for batch interpolation.
    
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
