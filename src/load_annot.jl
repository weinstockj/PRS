function load_annotations(annotations::String)

    annot = CSV.read(annotations, DataFrame)
    rename!(annot,:snp_id => :variant)

    return annot
end



