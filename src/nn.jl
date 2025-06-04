"""
    fit_genome_wide_nn(betas, annotation_files_dir, model_file; n_epochs, H, n_test, learning_rate_decay, patience)

Train a neural network model to predict variant effects using genome-wide annotation data.

# Arguments
- `betas`: Path to file containing PRS beta values from previous analysis
- `annotation_files_dir`: Directory containing annotation parquet files
- `model_file`: Path where the trained model will be saved
- `n_epochs`: Number of training epochs (default: 1306)
- `H`: Number of hidden units in the neural network (default: 3)
- `n_test`: Number of test samples to use for evaluation (default: 30)
- `learning_rate_decay`: Learning rate decay factor (default: 0.98)
- `patience`: Number of epochs to wait before reducing learning rate (default: 30)

# Returns
- `model`: The trained neural network model
- `opt`: The optimizer state
- `global_σ2`: Global residual variance estimate
"""
function fit_genome_wide_nn(
        betas = "/data/abattle4/april/hi_julia/prs_benchmark/prsfnn/jun22_adaptive_learning_rate/output/PRSFNN_out_final.tsv",
        annotation_files_dir = "/data/abattle4/jweins17/annotations/output/", 
        model_file = "trained_model.bson";
        n_epochs = 1306, H = 3, n_test = 30, learning_rate_decay = 0.98, patience = 30
    )

    summary_statistics = CSV.read(betas, DataFrame; delim = "\t")
    summary_statistics = rename!(summary_statistics, :variant => :variant_id)
    parquets = return_parquets(annotation_files_dir)

    required_columns = [:block, :block_residual_variance, :block_size]
    sigma_2s = unique(select(summary_statistics, required_columns))
    global_σ2 = calculate_global_residual_variance(sigma_2s)
    RSE = sqrt(global_σ2)
    @info "$(ltime()) Global residual variance: $(round(global_σ2; digits = 2)) and RSE: $(round(RSE; digits = 2))."

    # just to get K
    first_parquet = first(parquets)
    first_annot = select_annotation_columns(DataFrame(Parquet2.Dataset(first_parquet); copycols=false))
    K = size(first_annot, 2)

    layer_1 = Dense(K => H, Flux.softplus; init = Flux.glorot_normal(gain = 0.005))
    layer_output = Dense(H => 2)
    layer_output.bias .= [StatsFuns.log(0.001), StatsFuns.logit(0.1)]
    model = Chain(layer_1, layer_output)
    initial_lr = 0.00005
#    initial_lr = 0.0001
    optim_type = AdamW(initial_lr)
    opt = Flux.setup(optim_type, model)
    @info "$(ltime()) Training model with $K annotations"

    training_parquets = parquets[sample(1:length(parquets), n_epochs, replace = false)]
    n_epochs = length(training_parquets)
    test_parquets     = setdiff(parquets, training_parquets)[rand(1:(length(parquets) - n_epochs), n_test)]

    test_annotations = vcat([DataFrame(Parquet2.Dataset(x); copycols=false) for x in test_parquets]...)
    test_df        = innerjoin(test_annotations, summary_statistics, on = [:variant_id], makeunique=true)
    test_SNPs      = test_df.variant_id
    
    test_X = Float32.(transpose(select_annotation_columns(test_df[:, names(test_annotations)])))
    test_Y = Float32.(transpose(hcat(log.((test_df.mu .^ 2) ./ RSE), logit.(test_df.alpha))))
#    test_Y = Float32.(transpose(hcat(log.((test_df.mu .^ 2) ./ global_σ2), logit.(test_df.alpha))))

    @info "$(ltime()) Test set is comprised of $(length(test_parquets)) LD blocks and $(length(test_SNPs)) SNPs"

    best_loss = Inf
    count_since_best = 0
    best_model_epoch = 1
    train_losses = Float64[]
    test_losses = Float64[]

    best_model = deepcopy(model)
    best_opt = deepcopy(opt)

    @inbounds for i in 1:n_epochs
        annot_file = training_parquets[i]
        
        @info "$(ltime()) Epoch $i now reading $annot_file"
        annotations = DataFrame(Parquet2.Dataset(annot_file); copycols=false)
        # annotations = Parquet2.Dataset(annot_file)
        epoch_df = innerjoin(annotations, summary_statistics, on = [:variant_id])
        epoch_annot = select_annotation_columns(epoch_df[:, names(annotations)])

        X = Float32.(transpose(epoch_annot))
        Y = Float32.(transpose(hcat(log.((epoch_df.mu .^ 2 ./ RSE)), logit.(epoch_df.alpha))))
#        Y = Float32.(transpose(hcat(log.((epoch_df.mu .^ 2 ./ global_σ2)), logit.(epoch_df.alpha))))

        data = (X, Y)

        DL = Flux.DataLoader(data, batchsize=80, shuffle=true)

        train!(nn_loss, model, DL, opt)

        train_loss = nn_loss(
                model, 
                Float32.(X), 
                Float32.(Y)
            )

        push!(train_losses, train_loss)

        test_loss = nn_loss(
                model,
                test_X,
                test_Y
            )

        push!(test_losses, test_loss)

        if i == 1
            best_loss = test_loss
        end

        if test_loss < best_loss
#	if (best_loss - test_loss) / best_loss > 0.002
            best_loss = test_loss
            best_model = deepcopy(model)
	    best_opt = deepcopy(opt)
            best_model_epoch = i
            count_since_best = 0
        else
            count_since_best += 1
            Flux.adjust!(opt, opt.layers[1].weight.rule.opts[1].eta * learning_rate_decay)
        end

        if count_since_best >= patience
            @info "$(ltime()) Early stopping after $i epochs. "
            break
        end

        if i % 20 == 0
            GC.gc() # garbage collection every 20 epochs
        end

        @info "$(ltime()) Epoch $i training loss: $(round(train_loss; digits = 2)), test loss: $(round(test_loss; digits = 2)), best loss: $(round(best_loss; digits = 2)), count since best: $count_since_best"

    end

    original_params = Flux.params(model)
    copied_params = Flux.params(best_model)
    all_equal = all(map(==, original_params, copied_params))

    if all_equal
        @info "$(ltime()) Either training took place until the end or the best_model is not properly saved."
    end

    model = deepcopy(best_model)
    opt = deepcopy(best_opt)

    original_params = Flux.params(model)
    copied_params = Flux.params(best_model)
    all_equal = all(map(==, original_params, copied_params))

    if !all_equal
        @info "$(ltime()) Model being saved is not the best model."
    end

    @save model_file model opt
    
    write_global_residual_variance(betas, global_σ2)
    
    @info "$(ltime()) Best model and opt state taken from Epoch $best_model_epoch."

    return model, opt, train_losses, test_losses, setdiff(names(test_annotations), get_non_annotation_columns())
end
"""
 fit_heritability_nn(model, q_μ, q_μ_sq, q_alpha, G)

 Fit the heritability neural network model.

    # Arguments
    - `model::Chain`: A neural network model
    - `q_μ::Vector`: A length P vector of posterior means
    - `q_μ_sq::Vector`: A length P vector of posterior variances
    - `q_α::Vector`: A length P vector of posterior probabilities of being causal
    - `G::AbstractArray`: A P x K matrix of annotations

```julia-repl
    model = Chain(
        Dense(20 => 5, relu; init = Flux.glorot_normal(gain = 0.0005)),
        Dense(5 => 2)
    )

    G = rand(Normal(0, 1), 100, 20)
    q_μ_sq = (G * rand(Normal(0, 0.10), 20)) .^ 2
    q_α = 1.0 ./ (1.0 .+ exp.(-1.0 .* (-2.0 .+ q_μ_sq)))
    trained_model = PRSFNN.fit_heritability_nn(
        model, 
        q_μ_sq, 
        q_α, 
        G
    )
    yhat = transpose(trained_model(transpose(G)))
    yhat[:, 1] .= exp.(yhat[:, 1])
    yhat[:, 2] .= 1.0 ./ (1.0 .+ exp.(-yhat[:, 2]))
```
"""
function fit_heritability_nn(model, opt, q_μ_sq, q_α, G, i=1; max_epochs=3000, patience=100, mse_improvement_threshold=0.01, test_ratio=0.2, num_splits=10, learning_rate_decay = 0.95)


    # G_standardized = standardize(G)

    #λ = 1e-6
    println("----------------------------------------------------------------")
    @info "$(ltime()) Now training the neural network"
    # @info "$(ltime()) MIN FLOAT32 Q MU SQ: $(minimum(Float32.(q_μ_sq)))"
    # @info "$(ltime()) MAX FLOAT32 Q MU SQ: $(maximum(Float32.(q_μ_sq)))"
    q_μ_sq = clamp_nn_fit_h_nn(q_μ_sq)

    best_ks_statistic = Inf
    best_train_data = []
    best_test_data = []

    P = length(q_μ_sq)

    for _ in 1:num_splits
        # ak: shuffle indices
        permuted_indices = randperm(P)  
        # ak: get number of validation samples
        num_test = floor(Int, test_ratio * P)
        # ak: split into train and test based on above indices
        train_indices = permuted_indices[1:end-num_test]
        test_indices = permuted_indices[end-num_test+1:end]

        # ak: compute the KS statistic and pick the split with smallest KS stats
        ks_test = ApproximateTwoSampleKSTest(q_α[train_indices], q_α[test_indices])
        ks_n = ks_test.n_x*ks_test.n_y/(ks_test.n_x+ks_test.n_y)
        ks_statistic = (sqrt(ks_n)*ks_test.δ)

        if ks_statistic < best_ks_statistic
            best_ks_statistic = ks_statistic
            #best_train_data = [Float32.(G[train_indices, :]), Float32.(q_μ_sq[train_indices]), Float32.(q_α[train_indices])]
            best_train_data = [Float32.(G[train_indices, :]), Float32.(q_μ_sq[train_indices]), Float32.(q_α[train_indices])]
            #best_test_data = [Float32.(G[test_indices, :]), Float32.(q_μ_sq[test_indices]), Float32.(q_α[test_indices])]
            best_test_data = [Float32.(G[test_indices, :]), Float32.(q_μ_sq[test_indices]), Float32.(q_α[test_indices])]
        end

    end

    X = Float32.(transpose(best_train_data[1]))
    Y = Float32.(transpose(hcat(log.(best_train_data[2]), logit.(best_train_data[3]))))

    data = (X, Y)

    # Main.@infiltrate
    # DL = Flux.DataLoader(data, batchsize=10, shuffle=true, rng = Random.seed!(1))
    DL = Flux.DataLoader(data, batchsize=10, shuffle=true)
    
    best_loss = Inf
    best_model = deepcopy(model)
    best_opt = deepcopy(opt)
    count_since_best = 0
    best_model_epoch = 1
    train_losses = Float64[]
    test_losses = Float64[]

    @inbounds for epoch in 1:max_epochs
        train!(nn_loss, model, DL, opt)
        train_loss = nn_loss(
                model, 
                Float32.(transpose(best_train_data[1])), 
                Float32.(transpose(hcat(log.(best_train_data[2]), logit.(best_train_data[3]))))
            )
        push!(train_losses, train_loss)

        # ak: validation loss
        test_loss = nn_loss(
                model,
                Float32.(transpose(best_test_data[1])),
                Float32.(transpose(hcat(log.(best_test_data[2]), logit.(best_test_data[3]))))
            )
        push!(test_losses, test_loss)

        mse_improvement = (test_loss - best_loss) / test_loss

        if epoch % 50 == 0
            @info "$(ltime()) Epoch: $epoch, Train loss: $(round(train_loss, digits=3)), Test loss: $(round(test_loss, digits=3)), Relative change (ideally negative): $(round(mse_improvement; digits = 3))"
        end

        # if improvement from prev iteration is greater than threshold
        if mse_improvement < -mse_improvement_threshold
            best_loss = test_loss
            count_since_best = 0
            # set current epoch as to when best model was observed
            best_model_epoch = epoch
            # and save current model as best model
            best_model = deepcopy(model)
        else
            count_since_best += 1
            # @info "$(ltime()) Reducing learning rate to $(round(opt.layers[1].weight.rule.opts[1].eta * learning_rate_decay, digits = 5))"
            Flux.adjust!(opt, opt.layers[1].weight.rule.opts[1].eta * learning_rate_decay)
        end

        # If no improvement for "patience" epochs, stop training
        if count_since_best >= patience
            @info "$(ltime()) Early stopping after $epoch epochs. Now setting learning rate to 0.005"
            Flux.adjust!(opt, 0.005)
            break
        end

    end

    @info "$(ltime()) Done training. Best model taken from epoch $best_model_epoch."
    println("----------------------------------------------------------------")

    # plot_nn_loss = plot_loss_vs_epochs(train_losses, test_losses, i, best_model_epoch)
    # savefig("epoch_losses_$i.png")

    return best_model
end

function calculate_global_residual_variance(residual_variances)
    minimum_residual_variance = minimum(residual_variances.block_residual_variance)
    return minimum_residual_variance
end

function write_global_residual_variance(output_file, global_sigma2)
    output_file = split(output_file, ".")[1] * "_residual_variance.tsv"
    df = DataFrame(
        global_sigma2 = global_sigma2
    )
    CSV.write(output_file, df; delim = "\t")
end

function find_max_activation(layer, K)
    # create a set of one-hot encoded input patterns; 100 vectors
    # only one feature is active at a time
    dummy_inputs = [Float32.(I == j ? 1 : 0 for I in 1:K) for j in 1:K]

    # extract activation values of the neuron for each input pattern
    activations = [layer(input) for input in dummy_inputs]
    # Identify the input pattern that resulted in the maximum activation
    max_activations = [argmax(vec(activation)) for activation in activations]

    return max_activations
end

function predict_with_nn(model, G)
    outputs = model(transpose(G))
    nn_σ2_β = exp.(outputs[1, :])
    nn_p_causal = Flux.σ.(outputs[2, :]) 
    return nn_σ2_β, nn_p_causal
end

# RMSE
function nn_loss(model, G, y; w_σ2β = 1.0, w_p_causal = 1.0) ## ak: need two losses for slab variance and percent causal 

    yhat = model(G)
    σ2β_mse = @views Flux.mse(yhat[1, :], y[1, :])


    loss_σ2β = σ2β_mse 
    p_causal_mse = @views Flux.mse(yhat[2, :], y[2, :])

    loss_p_causal = p_causal_mse 

    return w_σ2β * loss_σ2β + w_p_causal * loss_p_causal ## ak: losses summed to form the total loss for training
end

invgamma_logpdf(x; α = 1000, θ = 1.0) = α * log(θ) - SpecialFunctions.loggamma(α) - (α + 1) * log(x) - θ / x
beta_logpdf(x; α = 1.0, β = 9.0) = xlogy(α - 1, x) + xlog1py(β - 1, -x) - SpecialFunctions.logbeta(α, β)

# function plot_loss_vs_epochs(train_losses, test_losses, i, epoch_model_taken)
#     plot(1:length(train_losses), train_losses, xlabel="Epochs", ylabel="Loss", label="train", title="Loss vs. Epochs, iteration $i, best at $epoch_model_taken", reuse=false)
#     plot!(1:length(test_losses), test_losses, lc=:orange, label="test")
#     vline!([epoch_model_taken], label="epoch of best")
# end

#function interpret_model(block = "chr18_59047676_60426196", model_file = "/data/abattle4/jweins17/PRS_runner/output/chr2/trained_model.bson", annot_data_path = "/data/abattle4/jweins17/annotations/output", gwas_file_name = "bmi_gwas.tsv", output_file = "effects.tsv"; min_MAF = 0.01)

#    annot_file = joinpath(annot_data_path, block, "variant_list_ccre_annotated_complete.parquet") #"variant_list_annotated_adult_fetal.bed")
#    #annot = CSV.read(annot_file, DataFrame)
#    annot = DataFrame(Parquet2.Dataset(annot_file); copycols=false)
#    rename!(annot,:variant_id => :SNP)

#    #cell_types = names(annot)[5:226]
#    #rename!(annot,:snp_id => :variant)
#    annotation_columns = names(annot)
#    #setdiff!(annotation_columns, ["snp_id", "chromosome", "start", "position"])
#    setdiff!(annotation_columns, ["chrom", "start", "end", "SNP", "ref", "alt"])
#    #push!(annotation_columns, "AF_ALT")
#    gwas_file = joinpath("/data/abattle4/april/hi_julia/annotations/ccre/celltypes", block, gwas_file_name)
#    annotations, summary_stats, current_LD_block_positions = load_annot_and_summary_stats(
#                annot_file,
#                #joinpath(annot_data_path, block, gwas_file_name);
#                gwas_file;
#		min_MAF = min_MAF
#            )

#    @load model_file model opt
#    nn_σ2_β, nn_p_causal = predict_with_nn(model, annotations)

#    max_p_causal_index = argmax(nn_p_causal)
#    annotations_copy = copy(annotations)
#    effects_σ2_β = zeros(length(annotation_columns))
#    effects_p_causal = zeros(length(annotation_columns))

#    for j in eachindex(annotation_columns)
#        annotations_copy[:, j] .= 1
#        nn_σ2_β_j, nn_p_causal_j = predict_with_nn(model, Float32.(annotations_copy))
#        effects_σ2_β[j] = nn_σ2_β_j[max_p_causal_index] - nn_σ2_β[max_p_causal_index]
#        effects_p_causal[j] = nn_p_causal_j[max_p_causal_index] - nn_p_causal[max_p_causal_index]
#        annotations_copy[:, j] = annotations[:, j] # reset the j-th column to the original values
#    end

#    df = DataFrame(cell_type = annotation_columns, effects_σ2_β = effects_σ2_β, effects_p_causal = effects_p_causal)

#    open(output_file, "w") do io
#        write(io, "cell_type\teffects_variance\teffects_PIP\n")
#        writedlm(io, [df.cell_type df.effects_σ2_β df.effects_p_causal], "\t")
#    end

#    return df
#end

