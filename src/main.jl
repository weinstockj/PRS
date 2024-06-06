"""

PRSFNN

This function defines the command line interface for the PRSFNN package.

# Arguments

- `summary_stats`: A path to an appropriately formatted summary statistics file
- `LD_reference`: A path a plink .bed file with the LD reference panel
- `annotations`: A path to an appropriately formatted annotations file

"""
@main function main(block::String = "chr18_12528622_15410468", 
            annot_data_path::String = "/data/abattle4/april/hi_julia/annotations/ccre/celltypes", 
            ld_panel_path::String = "/data/abattle4/jweins17/LD_REF_PANEL/output/bcf",
	    gwas_file_name::String = "bmi_gwas.tsv",
	    model_file::String = "trained_model.bson",
            betas_output_file::String = "PSRFNN_out.tsv", interpretation_output_file::String = "nn_interpretation.tsv"; min_MAF = 0.01, train_nn = true, H = 5, max_iter = 5)

    @info "$(ltime()) Current block: $block"
    current_chr = split(block, "_")[1]
    annotations, summary_stats, current_LD_block_positions = load_annot_and_summary_stats(
                joinpath(annot_data_path, block, "variant_list_annotated_adult_fetal.bed"),
                joinpath(annot_data_path, block, gwas_file_name);
                min_MAF = min_MAF
            )
    
    
    SNPs_count = size(annotations, 1)
    @info "$(ltime()) Number of SNPs in block: $SNPs_count"
    if isfile(model_file)
        @load model_file model
    else
        @info "$(ltime()) $model_file not found, creating new model!"
        # File doesn't exist, create a new model
        K = size(annotations, 2)
        layer_1 = Dense(K => H, relu; init = Flux.glorot_normal(gain = 0.0005))
        layer_output = Dense(H => 2)
        model = Chain(layer_1, layer_output)
    end

    mkpath(joinpath("data", block))
    LD_reference_filtered = joinpath("data", block, "filtered_EUR_current_block")
    snpdata = SnpData(joinpath(ld_panel_path, block, "filtered_EUR"))
    SnpArrays.filter(snpdata; des=LD_reference_filtered, f_snp = x -> x[:position] in current_LD_block_positions)
    LD_reference_filtered = LD_reference_filtered * ".bed"
    LD, D, good_variants = compute_LD(LD_reference_filtered)

    @assert !any(isnan.(LD))

    PRS = train_until_convergence(
        summary_stats.beta[good_variants],
        summary_stats.se[good_variants],
        LD, # correlation matrix already filtered for good variants
        D, # already filtered for good variants
        annotations[good_variants, :],
        model = model,
        N = summary_stats.n_complete_samples[good_variants],
        train_nn = train_nn,
        max_iter = max_iter
    )

    open(betas_output_file, "w") do io
        write(io, "variant\tmu\talpha\tvar\tss_beta\n")
        writedlm(io, [summary_stats.variant[good_variants] PRS[1] PRS[2] PRS[3] summary_stats.beta[good_variants]], "\t")
    end

    if train_nn
        model = PRS[4]
        @save model_file model
    end

    effects = interpret_model(
        block, 
        model_file, 
        annot_data_path, 
        gwas_file_name, 
        interpretation_output_file;
        min_MAF = min_MAF
    )
end


function interpret_model(block = "chr1_16103_1170341", model_file = "/data/abattle4/jweins17/PRS_runner/output/chr4/trained_model.bson", annot_data_path = "/data/abattle4/april/hi_julia/annotations/ccre/celltypes", gwas_file_name = "bmi_gwas.tsv", output_file = "effects.tsv"; min_MAF = 0.01)

    annot_file = joinpath(annot_data_path, block, "variant_list_annotated_adult_fetal.bed")
    annot = CSV.read(annot_file, DataFrame)
    cell_types = names(annot)[5:226]
    rename!(annot,:snp_id => :variant)
    annotations, summary_stats, current_LD_block_positions = load_annot_and_summary_stats(
                annot_file,
                joinpath(annot_data_path, block, gwas_file_name);
                min_MAF = min_MAF
            )

    @load model_file model
    nn_σ2_β, nn_p_causal = predict_with_nn(model, annotations)

    max_p_causal_index = argmax(nn_p_causal)
    annotations_copy = copy(annotations)
    effects_σ2_β = zeros(length(cell_types))
    effects_p_causal = zeros(length(cell_types))

    for j in eachindex(cell_types)
        annotations_copy[:, j] .= 1
        nn_σ2_β_j, nn_p_causal_j = predict_with_nn(model, Float32.(annotations_copy))
        effects_σ2_β[j] = nn_σ2_β_j[max_p_causal_index] - nn_σ2_β[max_p_causal_index]
        effects_p_causal[j] = nn_p_causal_j[max_p_causal_index] - nn_p_causal[max_p_causal_index]
        annotations_copy[:, j] = annotations[:, j] # reset the j-th column to the original values
    end

    df = DataFrame(cell_type = cell_types, effects_σ2_β = effects_σ2_β, effects_p_causal = effects_p_causal)

    open(output_file, "w") do io
        write(io, "cell_type\teffects_variance\teffects_PIP\n")
        writedlm(io, [df.cell_type df.effects_σ2_β df.effects_p_causal], "\t")
    end

    return df
end
