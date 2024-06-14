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

function clamp_ssr(ssr, max_value = 709.7) # slightly below the threshold
    return min.(ssr, max_value)
end

function clamp(x, ϵ = 1e-4)
    x = max.(min.(x, 1.0 - ϵ), ϵ) # to avoid Inf with logit transformation later
    return x
end

function logit(x)
    x = clamp(x)   
    return log.(x ./ (1 .- x))
end

function clamp_nn_fit_h_nn(q_μ_squared, max_value = 1e-5) # slightly below the threshold
    return max.(q_μ_squared, max_value)
end

