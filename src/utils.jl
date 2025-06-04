"""
    check_no_nan(data)

Verify that no NaN values are present in the first three elements of the input data.

# Arguments
- `data`: A collection (typically a tuple or array) where the first three elements are arrays to be checked for NaN values

# Throws
- `ErrorException`: If any NaN values are detected in the first three elements of `data`

# Details
This function is used as a validation check to ensure that critical data structures 
do not contain NaN values, which could cause numerical instability in downstream analyses.
"""
function check_no_nan(data)

    if sum(isnan.(data[1])) > 0
        error("NaN detected.")
    end
    
    if sum(isnan.(data[2])) > 0
        error("NaN detected.")
    end

    if sum(isnan.(data[3])) > 0
        error("NaN detected.")
    end
end

"""
    clamp_ssr(ssr, max_value = 709.7)

Clamp sum of squared residuals (SSR) values to prevent numerical overflow.

# Arguments
- `ssr`: Array or scalar of sum of squared residuals values
- `max_value`: Upper limit for clamping (default: 709.7, which is slightly below exp overflow threshold)

# Returns
- Clamped values, with no element greater than `max_value`

# Details
This function prevents numerical overflow in computations that involve exponentiation
of SSR values by limiting the maximum value to just below the overflow threshold for exp().
"""
function clamp_ssr(ssr, max_value = 709.7) # slightly below the threshold
    return min.(ssr, max_value)
end

"""
    clamp(x, ϵ = 1e-4)

Clamp values to the range [ϵ, 1-ϵ] to avoid numerical issues with logit transformations.

# Arguments
- `x`: Array or scalar of probability values to be clamped
- `ϵ`: Small positive number defining the margin from 0 and 1 (default: 1e-4)

# Returns
- Clamped values, with each element in the range [ϵ, 1-ϵ]

# Details
This function ensures that probability values are strictly between 0 and 1,
preventing Inf/-Inf results when applying logit transformations.
"""
function clamp(x, ϵ = 1e-4)
    x = max.(min.(x, 1.0 - ϵ), ϵ) # to avoid Inf with logit transformation later
    return x
end

"""
    logit(x)

Calculate the logit transformation of input values after clamping to avoid numerical issues.

# Arguments
- `x`: Array or scalar of probability values to transform

# Returns
- Logit-transformed values: log(x/(1-x)) for each element

# Details
The logit function is the inverse of the sigmoid function and maps probabilities
from [0,1] to [-∞,∞]. This implementation first clamps the input values to avoid
numerical issues and then applies the LogExpFunctions.logit function.
"""
function logit(x)
    x = clamp(x)   
    # return log.(x ./ (1 .- x))
    return LogExpFunctions.logit.(x)
end

"""
    clamp_nn_fit_h_nn(q_μ_squared, max_value = 1e-5)

Clamp squared mean values to a minimum threshold to ensure numerical stability in neural network fitting.

# Arguments
- `q_μ_squared`: Array or scalar of squared mean values
- `max_value`: Minimum allowed value (default: 1e-5)

# Returns
- Clamped values, with no element less than `max_value`

# Details
This function ensures that squared mean values used in neural network fitting
don't become too small, which could cause numerical instability.
"""
function clamp_nn_fit_h_nn(q_μ_squared, max_value = 1e-5) # slightly below the threshold
    return max.(q_μ_squared, max_value)
end

