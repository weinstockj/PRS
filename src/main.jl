"""

PRSFNN

This function defines the command line interface for the PRSFNN package.

# Arguments

- `summary_stats`: A path to an appropriately formatted summary statistics file
- `LD_reference`: A path a plink .bed file with the LD reference panel
- `annotations`: A path to an appropriately formatted annotations file

"""
@main function main(summary_stats::String, LD_reference::String, annotations::String;)

    #TODO: Complete definition of summary statistics, LD reference panel, and annotations

    # Load the summary statistics
    summary_stats = load_summary_stats(summary_stats)
    # Load the annotations
    annotations = load_annotations(annotations)
    # Subset for variants in summary statistics
    subset_annot_ss = innerjoin(annotations, summary_stats, on = [:variant], makeunique=true)
    annotations = Matrix(subset_annot_ss[:,5:226])
    summary_stats = subset_annot_ss[:,227:238] ## todo: change to using colnames

    current_LD_block_positions = subset_annot_ss[:,:position]

    # Load the LD reference panel
    LD = compute_LD(LD_reference)

    snpdata = SnpData(LD_reference)
    SnpArrays.filter(snpdata; des="test_data/current_block", f_snp = x -> x[:position] in current_LD_block_positions)
    LD, D = compute_LD("test_data/current_block.bed")

    # Run the PRS
    PRS = train_until_convergence(
            summary_stats.BETA,
            summary_stats.SE,
            # summary_stats.Z, 
            LD, # correlation matrix
            D,
            annotations
        )

end

