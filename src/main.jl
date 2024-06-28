"""

PRSFNN

This function defines the command line interface for the PRSFNN package.

# Arguments

- `output_prefix`: A prefix for the output files
- `annot_data_path`: A path to the directory containing the annotations
- `ld_panel_path`: A path to the directory containing the LD reference panel 
- `gwas_file_path`: A path to the GWAS summary statistics file 
- `model_file`: A path to the file containing the trained model 
- `betas_output_file`: A path to the file where the PRS betas will be saved 
- `interpretation_output_file`: A path to the file where the interpretation of the model will be saved

"""

@main function main(output_prefix::String = "chr13_110581699_111677479", 
            annot_data_path::String = "/data/abattle4/jweins17/annotations/output/chr13_110581699_111677479/variant_list_ccre_annotated_complete.parquet", 
            ld_panel_path::String = "/data/abattle4/jweins17/LD_REF_PANEL/output/bcf/chr13_110581699_111677479/filtered_EUR",

	    gwas_file_path::String = "/data/abattle4/april/hi_julia/annotations/ccre/celltypes/chr13_110581699_111677479/neale_bmi_gwas.tsv",
	    model_file::String = "trained_model.bson",
	    # model_file::String = "/data/abattle4/april/hi_julia/prs_benchmark/prsfnn/jun14_save_model_and_opt_state/output/chr13/trained_model.bson",
            betas_output_file::String = "PRSFNN_out.tsv", 
            interpretation_output_file::String = "nn_interpretation.tsv"; min_MAF = 0.01, train_nn = false, H = 5, max_iter = 5)

    @info "$(ltime()) Current block: $block"
    annotations, summary_stats, current_LD_block_positions = load_annot_and_summary_stats(
                annot_data_path,
                gwas_data_path,
                min_MAF = min_MAF
            )
    
    SNPs_count = size(annotations, 1)
    @info "$(ltime()) Number of SNPs in block: $SNPs_count"
    if isfile(model_file)
        @load model_file model opt
    else
        @info "$(ltime()) $model_file not found, creating new model!"
        # File doesn't exist, create a new model
        K = size(annotations, 2)
        layer_1 = Dense(K => H, Flux.softplus; init = Flux.glorot_normal(gain = 0.005))
        layer_output = Dense(H => 2)
        layer_output.bias .= [StatsFuns.log(0.0001), StatsFuns.logit(0.01)]
        model = Chain(layer_1, layer_output)
        initial_lr = 0.02
        optim_type = AdamW(initial_lr)
        opt = Flux.setup(optim_type, model)
    end

    LD_output_path = joinpath("LD_output", output_prefix)
    mkpath(LD_output_path)        

    LD_reference_filtered = joinpath(LD_output_path, "filtered")
    snpdata = SnpData(ld_panel_path)
    SnpArrays.filter(snpdata; des=LD_reference_filtered, f_snp = x -> x[:position] in current_LD_block_positions)
    LD_reference_filtered = LD_reference_filtered * ".bed"
    LD, D, good_variants = compute_LD(LD_reference_filtered)

    @assert !any(isnan.(LD))

    PRS = train_until_convergence(
        summary_stats.BETA[good_variants],
        summary_stats.SE[good_variants],
        LD, # correlation matrix already filtered for good variants
        D, # already filtered for good variants
        annotations[good_variants, :],
        model = model,
        opt = opt,
        N = summary_stats.N[good_variants],
        train_nn = train_nn,
        max_iter = max_iter
    )

    write_output_betas(betas_output_file, summary_stats, PRS, good_variants)

    if train_nn
        model = PRS[4]
        @save model_file model opt
    end

    # effects = interpret_model(
    #     block, 
    #     model_file, 
    #     annot_data_path, 
    #     gwas_file_name, 
    #     interpretation_output_file;
    #     min_MAF = min_MAF
    # )
end

function train_model(
        betas = "/data/abattle4/april/hi_julia/prs_benchmark/prsfnn/jun22_adaptive_learning_rate/output/PRSFNN_out_final.tsv",
        annotation_files_dir = "/data/abattle4/jweins17/annotations/output/", model_file = "trained_model.bson";
        n_epochs = 300, H = 3, n_test = 8, learning_rate_decay = 0.95, patience = 30
    )

    summary_statistics = CSV.read(betas, DataFrame)
    summary_statistics = rename!(summary_statistics, :variant => :variant_id)
    parquets = return_parquets(annotation_files_dir)

    # just to get K
    first_parquet = first(parquets)
    first_annot = select_annotation_columns(DataFrame(Parquet2.Dataset(first_parquet); copycols=false))
    K = size(first_annot, 2)

    layer_1 = Dense(K => H, Flux.softplus; init = Flux.glorot_normal(gain = 0.01))
    layer_output = Dense(H => 2)
    layer_output.bias .= [StatsFuns.log(0.0001), StatsFuns.logit(0.01)]
    model = Chain(layer_1, layer_output)
    initial_lr = 0.005
    optim_type = AdamW(initial_lr)
    opt = Flux.setup(optim_type, model)
    @info "$(ltime()) Training model with $K annotations"


    training_parquets = parquets[rand(1:length(parquets), n_epochs)]
    test_parquets     = setdiff(parquets, training_parquets)[rand(1:(length(parquets) - n_epochs), n_test)]

    @info "$(ltime()) Test set is comprised of $(length(test_parquets)) blocks"
    test_annotations = vcat([DataFrame(Parquet2.Dataset(x); copycols=false) for x in test_parquets]...)
    test_df        = innerjoin(test_annotations, summary_statistics, on = [:variant_id], makeunique=true)
    test_SNPs      = test_df.variant_id
    
    test_X = Float32.(transpose(select_annotation_columns(test_df[:, names(test_annotations)])))
    test_Y = Float32.(transpose(hcat(log.(test_df.mu .^ 2), logit.(test_df.alpha))))

    @info "$(ltime()) Test set is comprised of $(length(test_SNPs)) SNPs"

    # training_df  = summary_statistics[.!(in.(summary_statistics.variant_id, Ref(test_SNPs))), :]

    # @info "$(ltime()) Training set is comprised of $(size(training_df, 1)) SNPs"
    best_loss = Inf
    count_since_best = 0
    best_model_epoch = 1
    train_losses = Float64[]
    test_losses = Float64[]

    @inbounds for i in 1:n_epochs
        annot_file = training_parquets[i]
        
        @info "$(ltime()) Epoch $i now reading $annot_file"
        annotations = DataFrame(Parquet2.Dataset(annot_file); copycols=false)
        # annotations = Parquet2.Dataset(annot_file)
        epoch_df = innerjoin(annotations, summary_statistics, on = [:variant_id])
        epoch_annot = select_annotation_columns(epoch_df[:, names(annotations)])

        X = Float32.(transpose(epoch_annot))
        Y = Float32.(transpose(hcat(log.(epoch_df.mu .^ 2), logit.(epoch_df.alpha))))

        data = (X, Y)

        DL = Flux.DataLoader(data, batchsize=100, shuffle=true)

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

        if test_loss < best_loss
            best_loss = test_loss
            best_model = deepcopy(model)
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

        @info "$(ltime()) Epoch $i training loss: $train_loss, test loss: $test_loss, best loss: $best_loss, count since best: $count_since_best"

    end

    @save model_file model opt

    return model, opt, train_losses, test_losses, names(test_annotations)
end


function interpret_model(block = "chr18_59047676_60426196", model_file = "/data/abattle4/jweins17/PRS_runner/output/chr2/trained_model.bson", annot_data_path = "/data/abattle4/jweins17/annotations/output", gwas_file_name = "bmi_gwas.tsv", output_file = "effects.tsv"; min_MAF = 0.01)

    annot_file = joinpath(annot_data_path, block, "variant_list_ccre_annotated_complete.parquet") #"variant_list_annotated_adult_fetal.bed")
    #annot = CSV.read(annot_file, DataFrame)
    annot = DataFrame(Parquet2.Dataset(annot_file); copycols=false)
    rename!(annot,:variant_id => :SNP)

    #cell_types = names(annot)[5:226]
    #rename!(annot,:snp_id => :variant)
    annotation_columns = names(annot)
    #setdiff!(annotation_columns, ["snp_id", "chromosome", "start", "position"])
    setdiff!(annotation_columns, ["chrom", "start", "end", "SNP", "ref", "alt"])
    #push!(annotation_columns, "AF_ALT")
    gwas_file = joinpath("/data/abattle4/april/hi_julia/annotations/ccre/celltypes", block, gwas_file_name)
    annotations, summary_stats, current_LD_block_positions = load_annot_and_summary_stats(
                annot_file,
                #joinpath(annot_data_path, block, gwas_file_name);
                gwas_file;
		min_MAF = min_MAF
            )

    @load model_file model opt
    nn_σ2_β, nn_p_causal = predict_with_nn(model, annotations)

    max_p_causal_index = argmax(nn_p_causal)
    annotations_copy = copy(annotations)
    effects_σ2_β = zeros(length(annotation_columns))
    effects_p_causal = zeros(length(annotation_columns))

    for j in eachindex(annotation_columns)
        annotations_copy[:, j] .= 1
        nn_σ2_β_j, nn_p_causal_j = predict_with_nn(model, Float32.(annotations_copy))
        effects_σ2_β[j] = nn_σ2_β_j[max_p_causal_index] - nn_σ2_β[max_p_causal_index]
        effects_p_causal[j] = nn_p_causal_j[max_p_causal_index] - nn_p_causal[max_p_causal_index]
        annotations_copy[:, j] = annotations[:, j] # reset the j-th column to the original values
    end

    df = DataFrame(cell_type = annotation_columns, effects_σ2_β = effects_σ2_β, effects_p_causal = effects_p_causal)

    open(output_file, "w") do io
        write(io, "cell_type\teffects_variance\teffects_PIP\n")
        writedlm(io, [df.cell_type df.effects_σ2_β df.effects_p_causal], "\t")
    end

    return df
end

function write_output_betas(output_file, summary_stats, PRS, good_variants)

    df = DataFrame(
        variant = summary_stats.SNP[good_variants],
        mu = PRS[1],
        alpha = PRS[2],
        var = PRS[3],
        ss_beta = summary_stats.BETA[good_variants]
    )

    CSV.write(output_file, df)

    writefile(replace(output_file, "tsv" => "parquet", df)) # also write to parquet
end

function return_parquets(dir)

    all_parquets = []
    for (root, dirs, files) in walkdir(dir)
        for d in dirs
            sub_files = readdir(joinpath(root, d))
            parquet = filter(x -> occursin(".parquet", x), sub_files)
            push!(all_parquets, joinpath(root, d, first(parquet)))
        end
    end

    return all_parquets
end
