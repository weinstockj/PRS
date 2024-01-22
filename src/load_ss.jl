function load_summary_stats(summary_stats::String)

    function extract_chr_pos(variant_str)
        split_parts = split(variant_str, "_")
        return split_parts[1], parse(Int, split_parts[2])
    end

    unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

    ss = CSV.read(summary_stats, DataFrame)
    ss[!, :chromosome], ss[!, :position] = unzip(extract_chr_pos.(ss[:, :variant]))
    ss[!, :z] = ss[:, :beta] ./ ss[:, :se]
    ss = filter(row -> !row[:low_confidence_variant], ss)
    ss = subset(ss, :minor_AF => ByRow(>=(0.01)))

    return ss
end


