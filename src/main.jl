"""

PRSFNN

This function defines the command line interface for the PRSFNN package.

# Arguments

- `output_prefix`: A prefix for the output files
- `annot_data_path`: A path to the directory containing the annotations
- `ld_panel_path`: A path to the directory containing the LD reference panel 
- `gwas_data_path`: A path to the GWAS summary statistics file 
- `model_file`: A path to the file containing the trained model 
- `betas_output_file`: A path to the file where the PRS betas will be saved 
- `interpretation_output_file`: A path to the file where the interpretation of the model will be saved

"""

@main function main(output_prefix::String = "chr13_110581699_111677479", 
            annot_data_path::String = "/data/abattle4/jweins17/annotations/output/chr13_110581699_111677479/variant_list_ccre_annotated_complete.parquet", 
            ld_panel_path::String = "/data/abattle4/jweins17/LD_REF_PANEL/output/bcf/chr13_110581699_111677479/filtered_EUR",

	    gwas_data_path::String = "/data/abattle4/april/hi_julia/annotations/ccre/celltypes/chr13_110581699_111677479/neale_bmi_gwas.tsv",
	    # model_file::String = "/data/abattle4/april/hi_julia/prs_benchmark/prsfnn/jun14_save_model_and_opt_state/output/chr13/trained_model.bson",
            model_file::String = "trained_model.bson",
            betas_output_file::String = "PRSFNN_out.tsv", 
            interpretation_output_file::String = "nn_interpretation.tsv"; min_MAF = 0.01, train_nn = false, H = 5, max_iter = 5)

    @info "$(ltime()) Current block/output_prefix: $output_prefix"
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
        model = nothing
        opt   = nothing
    end

    LD_output_path = joinpath(output_prefix, "LD_output")
    @info "$(ltime()) Now creating directory $LD_output_path for LD output files."
    mkpath(LD_output_path)        

    LD_reference_filtered = joinpath(LD_output_path, "filtered")
    snpdata = SnpData(ld_panel_path)
    SnpArrays.filter(snpdata; des=LD_reference_filtered, f_snp = x -> x[:position] in current_LD_block_positions)
    LD_reference_filtered = LD_reference_filtered * ".bed"
    LD, X_sd, AF, good_variants = compute_LD(LD_reference_filtered)

    XtX = construct_XtX(LD, X_sd[good_variants], mean(summary_stats.N[good_variants]))
    D = construct_D(XtX)
    Xty = construct_Xty(summary_stats.BETA[good_variants], D)

    @assert !any(isnan.(LD))

    PRS = train_until_convergence(
        summary_stats.BETA[good_variants],
        summary_stats.SE[good_variants],
        LD, # correlation matrix already filtered for good variants
        XtX,
        Xty,
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
    #
    # return DataFrame(
    #     variant = summary_stats.SNP[good_variants],
    #     mu = PRS[1],
    #     alpha = PRS[2],
    #     var = PRS[3],
    #     ss_beta = summary_stats.BETA[good_variants],
    #     PVALUE = summary_stats.PVALUE[good_variants]
    # )
end

function write_output_betas(output_file, summary_stats, PRS, good_variants)

    df = DataFrame(
        variant = summary_stats.SNP[good_variants],
        mu = PRS[1],
        alpha = PRS[2],
        var = PRS[3],
        ss_beta = summary_stats.BETA[good_variants]
    )

    CSV.write(output_file, df; delim = "\t")

    Parquet2.writefile(replace(output_file, "tsv" => "parquet"), df) # also write to parquet
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
