This is a simple module which contains vectorised interpolators for linear and cubic-spline interpolation. It is compatible with torch's autograd. It extends torchinterp1d with Cubic Spline function with a few extrapolation options.


# History: 

- v0.1    11/7/2024 - Initial version, single function interpolation, both linear and cubic spline.
- v0.2    18/9/2024 - Implemented parallel handling of independent interpolations
          20/9/2024 - Fixed a forgotten contiguous conversion for already batched parameters
# Usage: 
Initialise the interpolation function with:

    f_int = interpolatorch.InterpolateLinear(x_vals, y_vals, extrapolate = False, ext=0, ext_value=None)

where `x_vals` and `y_vals` have shape `(N_b, N_t)`, with `N_b` counting the number of independent interpolations and `N_t` corresponds to the number of indices in each data set. `x_vals` needs to be sorted in dim=1. 

Then call with `f_int(x)` with any torch tensor `x`. If dim=0 of `x` has size `N_b`, then each element of these will be used to evaluate different interpolation functions. Otherwise, `x` will be assumed to apply to all interpolation functions. 

Same rules apply to `interpolatorch.CubicSplines`. 

If extrapolating:
- `ext = 0` : continuous extrapolation using the relationship at the closest boundary
- `ext = 1` : second order discontinuous extrapolation using the constant value at the closest boundary
- `ext = 2` : (potentially) first order discontinuous extrapolation using the values provided in `ext_value`.

# To do:
Currently, a single pair of extrapolation values are supported in `ext = 2` option. Separate pairs for each interpolation functions will be supported... if I need it.
