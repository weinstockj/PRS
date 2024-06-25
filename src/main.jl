"""

PRSFNN

This function defines the command line interface for the PRSFNN package.

# Arguments

- `summary_stats`: A path to an appropriately formatted summary statistics file
- `LD_reference`: A path a plink .bed file with the LD reference panel
- `annotations`: A path to an appropriately formatted annotations file

"""

@main function main(block::String = "chr13_110581699_111677479", #"chr2_10560_1415211", 
            annot_data_path::String = "/data/abattle4/jweins17/annotations/output", 
            ld_panel_path::String = "/data/abattle4/jweins17/LD_REF_PANEL/output/bcf",

	    gwas_file_name::String = "neale_bmi_gwas.tsv",
	    model_file::String = "trained_model.bson",
	    # model_file::String = "/data/abattle4/april/hi_julia/prs_benchmark/prsfnn/jun14_save_model_and_opt_state/output/chr13/trained_model.bson",
            betas_output_file::String = "PRSFNN_out.tsv", interpretation_output_file::String = "nn_interpretation.tsv"; min_MAF = 0.01, train_nn = true, H = 5, max_iter = 5)

    @info "$(ltime()) Current block: $block"
    current_chr = split(block, "_")[1]
    gwas_data_path = "/data/abattle4/april/hi_julia/annotations/ccre/celltypes"
    annotations, summary_stats, current_LD_block_positions = load_annot_and_summary_stats(
                joinpath(annot_data_path, block, "variant_list_ccre_annotated_complete.parquet"), #"variant_list_annotated_adult_fetal.bed"),
                joinpath(gwas_data_path, block, gwas_file_name);
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
        optim_type = AdamW(0.02)
        opt = Flux.setup(optim_type, model)
    end

    mkpath(joinpath("data", block))
    LD_reference_filtered = joinpath("data", block, "filtered_EUR_current_block")
    snpdata = SnpData(joinpath(ld_panel_path, block, "filtered_EUR"))
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

    open(betas_output_file, "w") do io
        write(io, "variant\tmu\talpha\tvar\tss_beta\n")
        writedlm(io, [summary_stats.SNP[good_variants] PRS[1] PRS[2] PRS[3] summary_stats.BETA[good_variants]], "\t")
    end

    if train_nn
        model = PRS[4]
        @save model_file model opt
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
