"""

PRSFNN

This function defines the command line interface for the PRSFNN package.

# Arguments

- `summary_stats`: A path to an appropriately formatted summary statistics file
- `LD_reference`: A path a plink .bed file with the LD reference panel
- `annotations`: A path to an appropriately formatted annotations file

"""
@main function main(block::String = "chr1_16103_1170341", 
            annot_data_path::String = "/data/abattle4/april/hi_julia/annotations/ccre/celltypes", 
            ld_panel_path::String = "/data/abattle4/jweins17/LD_REF_PANEL/output/bcf",
	    gwas_file_name::String = "bmi_gwas.tsv",
	    model_file::String = "trained_model.bson",
            output_file::String = "PSRFNN_out.tsv"; min_MAF = 0.01, train_nn = true, H = 5)

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
    LD_reference_filtered = joinpath("data", block, "filtered_current_block")
    snpdata = SnpData(joinpath(ld_panel_path, block, "filtered"))
    SnpArrays.filter(snpdata; des=LD_reference_filtered, f_snp = x -> x[:position] in current_LD_block_positions)
    LD_reference_filtered = LD_reference_filtered * ".bed"
    LD, D = compute_LD(LD_reference_filtered)

    PRS = train_until_convergence(
        summary_stats.beta,
        summary_stats.se,
        LD, # correlation matrix
        D,
        annotations,
        model = model,
        N = summary_stats.n_complete_samples,
        train_nn = train_nn
    )

    open(output_file, "w") do io
        write(io, "variant\tmu\talpha\tvar\tss_beta\n")
        writedlm(io, [summary_stats.variant PRS[1] PRS[2] PRS[3] summary_stats.beta], "\t")
    end

    if train_nn
        model = PRS[4]
        @save model_file model
    end
end

