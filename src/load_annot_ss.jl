function load_annot_and_summary_stats(annot::String, summary_statistics::String; min_MAF=0.01)
    
    unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

#    loading old annotation file (Zahng et al)
#    annot = CSV.read(annot, DataFrame)
#    rename!(annot,:snp_id => :SNP)

    annot = DataFrame(Parquet2.Dataset(annot); copycols=false)
    annot.ChIP = Int8.(annot[!, :ChIP])
    annot.Chromatin_accessibility = Int8.(annot[!, :Chromatin_accessibility])
    annot.QTL = Int8.(annot[!, :QTL])
    annot.PWM = Int8.(annot[!, :PWM])
    rename!(annot,:variant_id => :SNP)

    annotation_columns = names(annot)
    summary_statistics = CSV.read(summary_statistics, DataFrame)

    required_columns = [:SNP, :MAF, :N, :BETA, :SE, :PVALUE]
    summary_statistics = select(summary_statistics, required_columns)
    delete!(summary_statistics, findall(nonunique(select(summary_statistics, [:SNP]))))
    summary_statistics[!, :CHR], summary_statistics[!, :BP] = unzip(extract_chr_pos.(summary_statistics[:, :SNP]))
    # TODO: we should let the user decide on the min MAF
    summary_statistics = subset(summary_statistics, :MAF => ByRow(>=(min_MAF)))

    subset_annot_summary_statistics = innerjoin(annot, summary_statistics, on = [:SNP], makeunique=true)
    annotation_columns = names(subset_annot_summary_statistics)
    #println(annotation_columns)
    setdiff!(annotation_columns, ["chrom", "start", "end", "SNP", "ref", "alt", "SNP","MAF", "N", "BETA", "SE", "PVALUE", "CHR", "BP"])
    annot = Matrix(select(subset_annot_summary_statistics, annotation_columns))
    summary_statistics = select(subset_annot_summary_statistics, required_columns)
    current_LD_block_positions = subset_annot_summary_statistics[:,:BP]

    return annot, summary_statistics, current_LD_block_positions
end

function extract_chr_pos(variant_str)
    split_parts = split(variant_str, "_")
    return split_parts[1], parse(Int, split_parts[2])
end
