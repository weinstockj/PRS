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

    # Load the LD reference panel
    LD = compute_LD(LD_reference)

    # Load the annotations
    annotations = load_annotations(annotations)

    # Run the PRS
    PRS = train_until_convergence(
            summary_stats.BETA,
            summary_stats.SE,
            summary_stats.Z, 
            LD, # correlation matrix
            D,
            annotations
        )

end
