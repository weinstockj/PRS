"""

PRSFNN

This function defines the command line interface for the PRSFNN package.

# Arguments

- `summary_stats`: A path to an appropriately formatted summary statistics file
- `LD_reference`: A path a plink .bed file with the LD reference panel
- `annotations`: A path to an appropriately formatted annotations file

"""
#=
@main function main(summary_stats::String, LD_reference::String, annotations::String;)

    # Load the summary statistics
    summary_stats = load_summary_stats(summary_stats)
    # Load the annotations
    annotations = load_annotations(annotations)
    # Subset for variants in summary statistics
    subset_annot_ss = innerjoin(annotations, summary_stats, on = [:variant], makeunique=true)
    subset_annotations = Matrix(select(subset_annot_ss, 5:226))
    subset_summary_stats = select(subset_annot_ss, [4; 227:238]) ## change to using colnames
    current_LD_block_positions = subset_annot_ss[:,:position]

    # Load the LD reference panel
    snpdata = SnpData(LD_reference)
    SnpArrays.filter(snpdata; des="test_data/filtered_current_block", f_snp = x -> x[:position] in current_LD_block_positions)
    LD_reference_filtered = "test_data/filtered_current_block.bed"
    LD, D = compute_LD(LD_reference_filtered)

    # Run the PRS
    PRS = train_until_convergence(
            summary_stats.BETA,
            summary_stats.SE,
            LD, # correlation matrix
            D,
            annotations
        )

    PRS_out = DataFrame(mu = PRS[1], alpha = PRS[2], var = PRS[3])
end
=#


@main function main(summary_stats::String)
    summary_stats = load_summary_stats(summary_stats)
    #ld_block_names = "/home/akim126/data-abattle4/april/hi_julia/annotations/ccre/celltypes/ld_block_names.txt"
    ld_block_names = "/home/akim126/data-abattle4/april/hi_julia/annotations/ccre/celltypes/chr22_ld_block_names.txt"
    
    block_list = readlines("ld_block_names")

    for block in block_list
	@info "$(ltime()) Current block: $block"
        annotations = joinpath("/home/akim126/data-abattle4/april/hi_julia/annotations/ccre/celltypes", block, "variant_list_annotated_adult_fetal.bed")
        annotations = load_annotations(annotations)
        subset_annot_ss = innerjoin(annotations, summary_stats, on = [:variant], makeunique=true)
        subset_annotations = Matrix(select(subset_annot_ss, 5:226))
        subset_summary_stats = select(subset_annot_ss, [4; 227:239]) ## change to using colnames
        current_LD_block_positions = subset_annot_ss[:,:position]

        LD_reference = joinpath("/home/akim126/data-abattle4/jweins17/LD_REF_PANEL/output/bcf", block, "filtered")
        LD_reference_filtered = joinpath("data", block, "filtered_current_block")
        snpdata = SnpData(LD_reference)
        SnpArrays.filter(snpdata; des=LD_reference_filtered, f_snp = x -> x[:position] in current_LD_block_positions)
        LD_reference_filtered = LD_reference_filtered * ".bed"
        LD, D = compute_LD(LD_reference_filtered)

        PRS = train_until_convergence(
            subset_summary_stats.beta,
            subset_summary_stats.se,
            LD, # correlation matrix
            D,
            subset_annotations
        )

        PRS_out = DataFrame(variant = subset_summary_stats.variant, mu = PRS[1], alpha = PRS[2], var = PRS[3], ss_beta = subset_summary_stats.beta)
        save_output = joinpath("data", block, "prs_out.tsv")
        CSV.write(save_output, PRS_out)
    end
end
