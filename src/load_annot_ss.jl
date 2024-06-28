function load_annot_and_summary_stats(annotation_path::String, summary_statistics_path::String; min_MAF=0.01)
    
    annot = DataFrame(Parquet2.Dataset(annotation_path); copycols=false)
    annot.ChIP = Int8.(annot[!, :ChIP])
    annot.Chromatin_accessibility = Int8.(annot[!, :Chromatin_accessibility])
    annot.QTL = Int8.(annot[!, :QTL])
    annot.PWM = Int8.(annot[!, :PWM])
    rename!(annot,:variant_id => :SNP)

    annotation_columns = names(annot)
    summary_statistics = CSV.read(summary_statistics_path, DataFrame)

    required_columns = [:SNP, :MAF, :N, :BETA, :SE, :PVALUE]

    summary_statistics = select(summary_statistics, required_columns)
    delete!(summary_statistics, findall(nonunique(select(summary_statistics, [:SNP]))))
    summary_statistics[!, :CHR], summary_statistics[!, :BP] = unzip(extract_chr_pos.(summary_statistics[:, :SNP]))
    # TODO: we should let the user decide on the min MAF
    summary_statistics = subset(summary_statistics, :MAF => ByRow(>=(min_MAF)))

    subset_annot_summary_statistics = innerjoin(annot, summary_statistics, on = [:SNP], makeunique=true)

    annot = select_annotation_columns(subset_annot_summary_statistics)

    summary_statistics = select(subset_annot_summary_statistics, required_columns)
    current_LD_block_positions = subset_annot_summary_statistics[:,:BP]

    return annot, summary_statistics, current_LD_block_positions
end

function extract_chr_pos(variant_str)
    split_parts = split(variant_str, "_")
    return split_parts[1], parse(Int, split_parts[2])
end

unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

function get_non_annotation_columns()

    non_annotation_columns = ["chrom", "start", "end", "SNP", "ref", "alt", "SNP","MAF", "N", "BETA", "SE", "PVALUE", "CHR", "BP", "variant_id", "Standard"]

    return non_annotation_columns
end

function select_annotation_columns(df::DataFrame)

    all_columns = names(df)
    non_annotation_columns = get_non_annotation_columns()
    annotation_columns = setdiff(all_columns, non_annotation_columns)

    annotations = Matrix(select(df, annotation_columns))

    return annotations
end
