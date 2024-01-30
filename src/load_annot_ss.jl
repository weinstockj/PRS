function load_annot_and_summary_stats(annot::String, ss::String)

    function extract_chr_pos(variant_str)
        split_parts = split(variant_str, "_")
        return split_parts[1], parse(Int, split_parts[2])
    end

    unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

    annot = CSV.read(annot, DataFrame)
    rename!(annot,:snp_id => :variant)

    ss = CSV.read(ss, DataFrame)
    ss = select(ss, [:variant, :minor_AF, :low_confidence_variant, :beta, :se, :pval])
    ss[!, :chromosome], ss[!, :position] = unzip(extract_chr_pos.(ss[:, :variant]))
    # ss[!, :z] = ss[:, :beta] ./ ss[:, :se]
    ss = filter(row -> !row[:low_confidence_variant], ss)
    ss = subset(ss, :minor_AF => ByRow(>=(0.01)))

    subset_annot_ss = innerjoin(annot, ss, on = [:variant], makeunique=true)
    annot = Matrix(select(subset_annot_ss, 5:226))
    ss = select(subset_annot_ss, [:variant, :minor_AF, :low_confidence_variant, :beta, :se, :pval])
    current_LD_block_positions = subset_annot_ss[:,:position]

    return annot, ss, current_LD_block_positions
end


