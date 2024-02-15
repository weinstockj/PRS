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
	    gwas_file_name::String = "bmi_gwas.tsv";
	        model_file::String = "trained_model_locke_feb14.bson", H = 5, predict = false, ss_type="neale")

    @info "$(ltime()) Current block: $block"
    current_chr = split(block, "_")[1]
    annotations, summary_stats, current_LD_block_positions = load_annot_and_summary_stats(
                joinpath(annot_data_path, block, "variant_list_annotated_adult_fetal.bed"),
                joinpath(annot_data_path, block, gwas_file_name);
		ss_type
                #joinpath(annot_data_path, block, "bmi_gwas.tsv")
            )
    
    model_file = "data/" * current_chr * "_" * model_file
    
    if predict
        new_beta, new_se = update_gwas_effect_sizes(annotations, summary_stats, model_file)
        open(joinpath("data", block, "trained_updated_beta_se.tsv"), "w") do io
            write(io, "variant\tupdated_beta\tupdated_se\tss_beta\tss_se\n")
            writedlm(io, [summary_stats.variant new_beta new_se summary_stats.beta summary_stats.se], "\t")
        end
    
    else
        SNPs_count = size(annotations, 1)
        @info "$(ltime()) Number of SNPs in block: $SNPs_count"
        if isfile(model_file)
            println("file found")
            try
                @load model_file model
            println("printing model")
            println(model)
            catch e
                println("Error loading model from file:")
                println(e)
                # fallback to creating a new model
                K = size(annotations, 2)
                layer_1 = Dense(K => H, relu; init = Flux.glorot_normal(gain = 0.0005))
                layer_output = Dense(H => 2)
                model = Chain(layer_1, layer_output)
            println("loading failed; created model")
            println(model)
            end
        else
            println("file not found, creating new model!")
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

        @time PRS = train_until_convergence(
            summary_stats.beta,
            summary_stats.se,
            LD, # correlation matrix
            D,
            annotations,
            model=model,
            N=summary_stats.n_complete_samples
        )

        open(joinpath("data", block, "prs_out_locke_feb14.tsv"), "w") do io
            write(io, "variant\tmu\talpha\tvar\tss_beta\n")
            writedlm(io, [summary_stats.variant PRS[1] PRS[2] PRS[3] summary_stats.beta], "\t")
        end

        model = PRS[4]
        @save model_file model
    end
end

