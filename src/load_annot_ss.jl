function load_annot_and_summary_stats(annotation_path::String, summary_statistics_path::String; min_MAF=0.01)
    
    annot = DataFrame(Parquet2.Dataset(annotation_path); copycols=false)
    annot.ChIP = Int8.(annot[!, :ChIP])
    annot.Chromatin_accessibility = Int8.(annot[!, :Chromatin_accessibility])
    annot.QTL = Int8.(annot[!, :QTL])
    annot.PWM = Int8.(annot[!, :PWM])
    rename!(annot,:variant_id => :SNP)

    annotation_columns = names(annot)
    summary_statistics = CSV.read(summary_statistics_path, DataFrame)

    # rename!(
    #     summary_statistics, 
    #     :variant => :SNP,
    #     :minor_AF => :MAF,
    #     :n_complete_samples => :N,
    #     :beta => :BETA,
    #     :se => :SE,
    #     :pval => :PVALUE
    # )

    # required_columns = [:SNP, :MAF, :N, :BETA, :SE, :PVALUE, :ytx]
    required_columns = [:SNP, :MAF, :N, :BETA, :SE, :PVALUE]

    summary_statistics = select(summary_statistics, required_columns)
    summary_statistics.SNP = String.(summary_statistics.SNP)

    @info "$(ltime()) Now standardizing beta"
    summary_statistics.BETA_std = standardize_beta(
                summary_statistics.BETA, 
                summary_statistics.SE, 
                summary_statistics.N,
                summary_statistics.MAF
    )

    push!(required_columns, :BETA_std)

    # @info "$(ltime()) here 1"
    delete!(summary_statistics, findall(nonunique(select(summary_statistics, [:SNP]))))
    # @info "$(ltime()) here 2"
    summary_statistics[!, :CHR], summary_statistics[!, :BP] = unzip(extract_chr_pos.(summary_statistics[:, :SNP]))
    # TODO: we should let the user decide on the min MAF
    summary_statistics = subset(summary_statistics, :MAF => ByRow(>=(min_MAF)))
    # @info "$(ltime()) here 3"

    subset_annot_summary_statistics = innerjoin(annot, summary_statistics, on = [:SNP], makeunique=true)
    # @info "$(ltime()) here 4"

    subset_annot_summary_statistics = unique(subset_annot_summary_statistics, :SNP) # takes first occurrence if multiple rows present for each SNP

    annot = select_annotation_columns(subset_annot_summary_statistics)
    # @info "$(ltime()) here 5"

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

    non_annotation_columns = ["chrom", "start", "end", "SNP", "ref", "alt", "SNP","MAF", "N", "BETA", "SE", "PVALUE", "CHR", "BP", "variant_id", "Standard", "BETA_std"]

    return non_annotation_columns
end

function select_annotation_columns(df::DataFrame)

    all_columns = names(df)
    non_annotation_columns = get_non_annotation_columns()
    annotation_columns = setdiff(all_columns, non_annotation_columns)

    annotations = Matrix(select(df, annotation_columns))

    return annotations
end

function standardize_beta(BETA::Vector{Float64}, SE::Vector{Float64}, N::Vector{Int64}, MAF::Vector{Float64})
    σ2y = median(2 .* MAF .* (1 .- MAF) .* (N .* (SE .^ 2) .+ BETA .^ 2))
    s = sqrt.((σ2y ./ (N .* SE .^ 2 .+ BETA .^ 2)))
    return BETA .* s
end
