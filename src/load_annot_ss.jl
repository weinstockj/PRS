function load_annot_and_summary_stats(annot::String, summary_statistics::String; min_MAF=0.01)

    unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

    annot = CSV.read(annot, DataFrame)
    rename!(annot,:snp_id => :variant)

    summary_statistics = CSV.read(summary_statistics, DataFrame)

    # TODO: we need to decide on a standard input format
    required_columns = [:variant, :minor_AF, :n_complete_samples, :beta, :se, :pval]
    summary_statistics = select(summary_statistics, required_columns)
    delete!(summary_statistics, findall(nonunique(select(summary_statistics, [:variant]))))
    summary_statistics[!, :chromosome], summary_statistics[!, :position] = unzip(extract_chr_pos.(summary_statistics[:, :variant]))
    # TODO: we should let the user decide on the min MAF
    summary_statistics = subset(summary_statistics, :minor_AF => ByRow(>=(min_MAF)))

    subset_annot_summary_statistics = innerjoin(annot, summary_statistics, on = [:variant], makeunique=true)
    # TODO: hardcoding the columns to be selected is not ideal
    annot = Matrix(select(subset_annot_summary_statistics, 5:226))
    summary_statistics = select(subset_annot_summary_statistics, required_columns) 
    current_LD_block_positions = subset_annot_summary_statistics[:,:position]

    return annot, summary_statistics, current_LD_block_positions
end

function extract_chr_pos(variant_str)
    split_parts = split(variant_str, "_")
    return split_parts[1], parse(Int, split_parts[2])
end
