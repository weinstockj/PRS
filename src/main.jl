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
@main function main(
            output_prefix::String = "chr3_175214913_176977984", 
            annot_data_path::String = "/data/abattle4/jweins17/annotations/output/chr3_175214913_176977984/variant_list_ccre_annotated_complete.parquet", 
            ld_panel_path::String = "/data/abattle4/jweins17/LD_REF_PANEL/output/bcf/chr3_175214913_176977984/filtered_EUR",
	        gwas_data_path::String = "/data/abattle4/april/hi_julia/annotations/ccre/celltypes/chr3_175214913_176977984/neale_bmi_gwas.tsv",
            model_file::String = "",
            betas_output_file::String = "PRSFNN_out_cavi.tsv", 
            interpretation_output_file::String = "nn_interpretation.tsv",
            first_stage_rv_file::String = "PRSFNN_out_initial.tsv"; min_MAF = 0.01, train_nn = false, H = 5, max_iter = 5)

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
    LD_reference_filtered_bed = LD_reference_filtered * ".bed"
    LD, X_sd, AF, good_variants = compute_LD(LD_reference_filtered_bed)

    LD_SNPs = CSV.read(LD_reference_filtered * ".bim", DataFrame; header = false)
    good_LD_SNPs = LD_SNPs.Column2[good_variants]
    intersect_SNPs = intersect(good_LD_SNPs, summary_stats.SNP)

    good_indices = findall(in.(summary_stats.SNP, Ref(intersect_SNPs)))
    summary_stats = summary_stats[good_indices, :]
    annotations = annotations[good_indices, :]

    @assert nrow(summary_stats) == length(intersect_SNPs)
    @assert isequal(summary_stats.SNP, good_LD_SNPs)

    XtX = construct_XtX(LD, X_sd[good_variants], mean(summary_stats.N))
    D = construct_D(XtX)
    Xty = construct_Xty(summary_stats.BETA, D)

    σ2, R2, yty = infer_σ2(
        summary_stats.BETA, 
        summary_stats.SE, 
        XtX, 
        Xty, 
        median(summary_stats.N), 
        length(summary_stats.BETA); 
        estimate = true, 
        λ = 0.50 * median(summary_stats.N)
    )

    XtX .= construct_XtX(LD, ones(length(summary_stats.SNP)), mean(summary_stats.N))
    D .= construct_D(XtX)
    Xty .= construct_Xty(summary_stats.BETA_std, D)

    @assert !any(isnan.(LD))

    if isfile(first_stage_rv_file)
        calculated_σ2 = CSV.read(first_stage_rv_file, DataFrame)
        σ2 = calculated_σ2.global_sigma2[1]       
        @info "$(ltime()) Calculating and passing global residual variance $(round(σ2; digits = 2)) after NN tratining."
    end

    PRS = train_until_convergence(
        summary_stats.BETA_std,
        summary_stats.SE,
        LD, # correlation matrix already filtered for good variants
        XtX,
        Xty,
        annotations,
        model = model,
        opt = opt,
        σ2 = σ2,
        R2 = R2,
        yty = yty,
        N = summary_stats.N,
        train_nn = train_nn,
        max_iter = max_iter
    )

    write_output_betas(betas_output_file, summary_stats, PRS, output_prefix)
    # write_output_residual_variance(betas_output_file, PRS, output_prefix)

    if train_nn
        error("This code path is currently out of date and needs to be updated.")
        model = PRS[7]
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
    return PRS
end

"""
Write the PRS beta coefficients and related statistics to output files.

# Arguments
- `output_file::String`: Path where the output file will be saved
- `summary_stats::DataFrame`: DataFrame containing summary statistics from GWAS
- `PRS::Tuple`: Tuple containing PRS calculation results including:
    - PRS[1]: Mean effects (μ)
    - PRS[2]: Alpha values
    - PRS[3]: Spike means
    - PRS[4]: Neural network sigma beta values
    - PRS[5]: Neural network causal probabilities
    - PRS[6]: Block residual variance
- `ld_block_name::String`: Name of the LD block being processed

# Outputs
Saves two files:
- A TSV file at the specified output path
- A Parquet file with the same name but .parquet extension
"""
function write_output_betas(output_file, summary_stats, PRS, ld_block_name)

    df = DataFrame(
        variant = summary_stats.SNP,
        mu = PRS[1],
        alpha = PRS[2],
        mu_spike = PRS[3],
        ss_beta = summary_stats.BETA,
	    nn_sigma_beta = PRS[4],
	    nn_p_causal = PRS[5],
        block = ld_block_name,
        block_residual_variance = PRS[6],
        block_size = length(PRS[1]),
    )

    CSV.write(output_file, df; delim = "\t")

    Parquet2.writefile(replace(output_file, "tsv" => "parquet"), df) # also write to parquet
end

"""
Find all parquet files within the subdirectories of a specified directory.

# Arguments
- `dir::String`: The directory to search for parquet files

# Returns
- `Vector{String}`: A vector of absolute paths to the parquet files found

# Details
This function walks through the directory structure, looking for .parquet files
in each subdirectory. It assumes that each directory contains at most one parquet file,
and returns the path to that file.
"""
function return_parquets(dir)


    all_parquets = Vector{String}()
    for (root, dirs, files) in walkdir(dir)
        for d in dirs
            sub_files = readdir(joinpath(root, d))
            parquet = filter(x -> occursin(".parquet", x), sub_files)
            push!(all_parquets, joinpath(root, d, first(parquet)))
        end
    end

    return all_parquets
end

"""
Write block-level residual variance information to a TSV file.

# Arguments
- `output_file::String`: Base file path, which will be modified to create the residual variance file path
- `PRS::Tuple`: Tuple containing PRS calculation results, where PRS[6] contains block residual variance
- `ld_block_name::String`: Name of the LD block being processed

# Outputs
Creates a TSV file with the residual variance information, with filename derived from the input file
by replacing its extension with "residual_variance.tsv".

# Details
The output file contains three columns: block name, residual variance, and block size.
"""
function write_output_residual_variance(output_file, PRS, ld_block_name)


    output_file = split(output_file, ".")[1] * "residual_variance.tsv"

    df = DataFrame(
        block = ld_block_name,
        block_residual_variance = PRS[6],
        block_size = length(PRS[1])
    )

    CSV.write(output_file, df; delim = "\t")
    
end


