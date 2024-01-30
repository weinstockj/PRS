"""

PRSFNN

This function defines the command line interface for the PRSFNN package.

# Arguments

- `summary_stats`: A path to an appropriately formatted summary statistics file
- `LD_reference`: A path a plink .bed file with the LD reference panel
- `annotations`: A path to an appropriately formatted annotations file

"""
@main function main(block::String, data_path::String, ld_panel_path::String)
    # data_path: "/annotations/ccre/celltypes"
    # ld_panel_path: "LD_REF_PANEL/.."

    @info "$(ltime()) Current block: $block"
    annotations, summary_stats, current_LD_block_positions = load_annot_and_summary_stats(joinpath(data_path, block, "variant_list_annotated_adult_fetal.bed"), joinpath(data_path, block, "bmi_gwas.tsv"))

    mkpath(joinpath("data", block))
    LD_reference_filtered = joinpath("data", block, "filtered_current_block")
    snpdata = SnpData(joinpath(ld_panel_path, block, "filtered"))
    SnpArrays.filter(snpdata; des=LD_reference_filtered, f_snp = x -> x[:position] in current_LD_block_positions)
    LD_reference_filtered = LD_reference_filtered * ".bed"
    LD, D = compute_LD(LD_reference_filtered)

    @time PRS = train_until_convergence!(
    # PRS = train_until_convergence!(
        summary_stats.beta,
        summary_stats.se,
        LD, # correlation matrix
        D,
        annotations
    )

    open(joinpath("data", block, "prs_out_0129_testing.tsv"), "w") do io
        write(io, "variant\tmu\talpha\tvar\tss_beta\n")
        writedlm(io, [summary_stats.variant PRS[1] PRS[2] PRS[3] summary_stats.beta], "\t")
    end
end




#=
    data_path = "/home/akim126/data-abattle4/april/hi_julia/annotations/ccre/celltypes"
    # block_list = readlines(ld_block_names)

    @time begin
        for block in readlines(ld_block_names)
    	    @info "$(ltime()) Current block: $block"
            @time annotations, summary_stats, current_LD_block_positions = load_annot_and_summary_stats(joinpath(data_path, block, "variant_list_annotated_adult_fetal.bed"), joinpath(data_path, block, "bmi_gwas.tsv"))

	    mkpath(joinpath("data", block))
	    LD_reference_filtered = joinpath("data", block, "filtered_current_block")
            snpdata = SnpData(joinpath("/home/akim126/data-abattle4/jweins17/LD_REF_PANEL/output/bcf", block, "filtered"))
            SnpArrays.filter(snpdata; des=LD_reference_filtered, f_snp = x -> x[:position] in current_LD_block_positions)
            LD_reference_filtered = LD_reference_filtered * ".bed"
            @time LD, D = compute_LD(LD_reference_filtered)

	    rm(LD_reference_filtered * ".bed", force=true)
	    rm(LD_reference_filtered * ".bim", force=true)
	    rm(LD_reference_filtered * ".fam", force=true)
        
	    PRS = train_until_convergence!(
                summary_stats.beta,
                summary_stats.se,
                LD, # correlation matrix
                D,
                annotations
            )

	    open(joinpath("data", block, "prs_out_0129.tsv"), "w") do io
	        write(io, "variant\tmu\talpha\tvar\tss_beta\n")
	        writedlm(io, [summary_stats.variant PRS[1] PRS[2] PRS[3] summary_stats.beta], "\t")
	    end
        end
    end
end
=#
