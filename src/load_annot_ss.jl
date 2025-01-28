function load_annot_and_summary_stats(annotation_path::String, summary_statistics_path::String; min_MAF=0.01)
    
    annot = DataFrame(Parquet2.Dataset(annotation_path); copycols=false)
#    annot = sort(annot, :start)
#    annot.ChIP = Int8.(annot[!, :ChIP])
#    annot.Chromatin_accessibility = Int8.(annot[!, :Chromatin_accessibility])
#    annot.QTL = Int8.(annot[!, :QTL])
#    annot.PWM = Int8.(annot[!, :PWM])
    rename!(annot,:variant_id => :SNP)

#    annotation_columns = names(annot)
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

    non_annotation_columns = ["chrom", "start", "end", "SNP", "ref", "alt", "SNP","MAF", "N", "BETA", "SE", "PVALUE", "CHR", "BP", "variant_id", "Standard", "BETA_std", "mu", "alpha", "mu_spike", "ss_beta", "nn_sigma_beta", "nn_p_causal", "block", "block_residual_variance", "block_size"]

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

function get_all_annot_columns()

    new_annotation_columns = ["Myoepithelial", "Keratinocyte_1", "Sm_Ms_Mucosal", "Skin_Granular_Epidermal", "Memory_B", "Alveolar_Type_2_Immune", "Type_II_Skeletal_Myocyte", "Endothelial_General_3", "Sm_Ms_GE_junction", "Plasma_B", "Schwann_General", "Tuft", "Parietal", "Colon_Epithelial_1", "T_lymphocyte_2_CD4+", "BBB_Endothelial", "Mammary_Epithelial", "Sm_Ms_Muscularis_3", "Pericyte_General_1", "Glomerulosa", "SI_Goblet", "A_Cardiomyocyte", "Fibro_Epithelial", "Lymphatic", "Enteric_Neuron", "Cortical_Epithelial", "Mammary_Luminal_Epi_1", "Enterocyte", "Pericyte_General_2", "Sm_Ms_Uterine", "Colon_Epithelial_2", "Fibro_Nerve", "Pericyte_Muscularis", "Nerve_Stromal", "Luteal", "Mammary_Luminal_Epi_2", "Sm_Ms_Muscularis_2", "Oligo_Precursor", "Mesothelial", "Sm_Ms_Muscularis_1", "Delta+Gamma", "Alpha_1", "T_Lymphocyte_1_CD8+", "Fasciculata", "Enterochromaffin", "Paneth", "Alveolar_Cap_Endo", "Endothelial_General_1", "Sm_Ms_Colon_1", "Endothelial_General_2", "Astrocyte_1", "Endothelial_Exocrine", "Skin_Basal_Epidermal", "Skin_Eccrine_Epidermal", "Glutamatergic_1", "Macrophage_Gen_or_Alv", "Sm_Ms_Vaginal", "Mammary_Basal_Epi", "Cardiac_Pericyte_2", "Microglia", "Glutamatergic_2", "Endothelial_Myocardial", "Colon_Epithelial_3", "Adipocyte", "Cardiac_Pericyte_1", "Pericyte_General_4", "Sm_Ms_Colon_2", "Fibro_General", "Beta_1", "Naive_T", "Chief", "Satellite", "Colon_Goblet", "Vasc_Sm_Muscle_1", "Oligodendrocyte", "Macrophage_General", "Club", "Fibro_GI", "Pericyte_General_3", "Acinar", "V_Cardiomyocyte", "Fibro_Liver_Adrenal", "Alveolar_Type_1", "Natural_Killer_T", "Cardiac_Pericyte_4", "Follicular", "Sm_Ms_GI", "Type_I_Skeletal_Myocyte", "Fibro_Muscle", "Mast", "Astrocyte_2", "Cardiac_Fibroblast", "Cardiac_Pericyte_3", "Sm_Ms_General", "Gastric_Neuroendocrine", "Ductal", "Keratinocyte_2", "Esophageal_Epithelial", "Transitional_Cortical", "Hepatocyte", "GABA_1", "Beta_2", "Alpha_2", "Alveolar_Type_2", "Endocardial", "Cilliated", "Vasc_Sm_Muscle_2", "Foveolar", "Airway_Goblet", "GABA_2", "Melanocyte", "Fetal_Excitatory_Neuron_12", "Fetal_Enteroendocrine", "Fetal_Astrocyte_3", "Fetal_Inhibitory_Neuron_2", "Fetal_Astrocyte_1", "Fetal_Macrophage_General_1", "Fetal_Lymphatic", "Fetal_Inhibitory_Neuron_3", "Fetal_T_Lymphocyte_1_CD4+", "Fetal_Erythroblast_1", "Fetal_Endothelial_General_2", "Fetal_Alveolar_Epithelial_2", "Fetal_Goblet", "Fetal_Enterocyte_1", "Fetal_Inhibitory_Neuron_4", "Fetal_Satellite_2", "Fetal_B_Lymphocyte_3_NPY+", "Fetal_A_Cardiomyocyte", "Fetal_Stellate", "Fetal_B_Lymphocyte_2_CXCR5+", "Fetal_Astrocyte_5", "Fetal_Macrophage_Hepatic_2", "Fetal_Fibro_Splenic", "Fetal_Hepatoblast", "Fetal_T_Lymphocyte_4_FASLG+", "Fetal_Cardiac_Fibroblast", "Fetal_Parietal+Chief", "Fetal_Fibro_Muscle_1", "Fetal_Thymocyte", "Fetal_Excitatory_Neuron_8", "Fetal_Endothelial_General_3", "Fetal_Skeletal_Myocyte_3", "Fetal_Skeletal_Myocyte_2", "Fetal_T_Lymphocyte_2_Cytotoxic", "Fetal_Astrocyte_4", "Fetal_Chromaffin", "Fetal_Endocardial", "Fetal_Photoreceptor", "Fetal_Macrophage_Hepatic_3", "Fetal_Acinar_2", "Fetal_Fibro_General_2", "Fetal_Retinal_Pigment", "Fetal_Sympathoblast", "Fetal_Erythroblast_4", "Fetal_Endothelial_General_1", "Fetal_Excitatory_Neuron_11", "Fetal_Endothelial_Hepatic_1", "Fetal_Astrocyte_2", "Fetal_Excitatory_Neuron_2", "Fetal_Hematopoeitic_Stem", "Fetal_Cholangiocyte", "Fetal_Erythroblast_2", "Fetal_Metanephric", "Fetal_Schwann_General", "Fetal_Pulmonary_Neuroendocrine", "Fetal_Enterocyte_3", "Fetal_Mesangial_2", "Fetal_Endothelial_Hepatic_2", "Fetal_Excitatory_Neuron_6", "Fetal_Oligo_Progenitor_2", "Fetal_T_Lymphocyte_3_IL2+", "Fetal_Excitatory_Neuron_10", "Fetal_Retinal_Progenitor_1", "Fetal_Mesangial_1", "Fetal_Retinal_Progenitor_2", "Fetal_Inhibitory_Neuron_5", "Fetal_Inhibitory_Neuron_1", "Fetal_Erythroblast_5", "Fetal_Macrophage_General_4", "Fetal_Mesothelial", "Fetal_V_Cardiomyocyte", "Fetal_Retinal_Neuron", "Fetal_Enteric_Glia", "Fetal_Extravillous_Trophoblast", "Fetal_Adrenal_Neuron", "Fetal_Excitatory_Neuron_4", "Fetal_B_Lymphocyte_1_SPIB+", "Fetal_Excitatory_Neuron_9", "Fetal_Enteric_Neuron", "Fetal_Cilliated", "Fetal_Alveolar_Epithelial_1", "Fetal_Macrophage_Placental", "Fetal_Macrophage_General_3", "Fetal_Excitatory_Neuron_3", "Fetal_Macrophage_Hepatic_1", "Fetal_Fibro_GI", "Fetal_Endothelial_Placental", "Fetal_Gastri_Goblet", "Fetal_Erythroblast_3", "Fetal_Macrophage_General_2", "Fetal_Fibro_Placental_1", "Fetal_Fibro_General_5", "Fetal_Skeletal_Myocyte_1", "Fetal_Placental_Neuron", "Fetal_Syncitio+Cytotrophoblast", "Fetal_Excitatory_Neuron_7", "Fetal_Satellite_1", "Fetal_Acinar_1", "Fetal_Fibro_General_3", "Fetal_Islet", "Fetal_Megakaryocyte", "Fetal_Fibro_General_1", "Fetal_Enterocyte_2", "Fetal_Excitatory_Neuron_1", "Fetal_Ureteric_Bud", "Fetal_Fibro_Placental_2", "Fetal_Fibro_General_4", "Fetal_Alveolar_Cap_Endo", "Fetal_Excitatory_Neuron_5", "Fetal_Ductal", "Fetal_Adrenal_Cortical", "RoCC", "UCE", "AF_EUR", "AF_AFR", "AF_AMR", "AF_EAS", "AF_SAS", "ChIP", "Chromatin_accessibility", "QTL", "PWM", "am_pathogenicity", "CA-TF", "CA", "dELS", "CA-CTCF", "TF", "pELS", "CA-H3K4me3", "PLS"]

    return new_annotation_columns
end

function fill_in_missing_annot_cols(df)
    new_annotation_columns = get_all_annot_columns()
    df = sort(df, :start)
    missing_cols = setdiff(new_annotation_columns, names(df))
    for col in missing_cols
       df[!,col] .= 0.0
#       df[!,col] .= missing
    end
    select!(df, :chrom, :start, :end, :variant_id, :ref, :alt, new_annotation_columns)
    return df
end
