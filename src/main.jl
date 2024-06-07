"""

PRSFNN

This function defines the command line interface for the PRSFNN package.

# Arguments

- `summary_stats`: A path to an appropriately formatted summary statistics file
- `LD_reference`: A path a plink .bed file with the LD reference panel
- `annotations`: A path to an appropriately formatted annotations file

"""
@main function main(block::String = "chr1_16103_1170341", #"chr15_23850554_26295433", 
            annot_data_path::String = "/data/abattle4/april/hi_julia/annotations/ccre/celltypes", 
            ld_panel_path::String = "/data/abattle4/jweins17/LD_REF_PANEL/output/bcf",
	    #gwas_file_name::String = "bmi_gwas.tsv",
	    #gwas_file_name::String = "ukbb_409k_bmi_gwas_cp.tsv",
	    gwas_file_name::String = "neale_bmi_gwas.tsv",
	    model_file::String = "trained_model.bson",
            betas_output_file::String = "PSRFNN_out.tsv", interpretation_output_file::String = "nn_interpretation.tsv"; min_MAF = 0.01, train_nn = true, H = 5, max_iter = 2)

    @info "$(ltime()) Current block: $block"
    current_chr = split(block, "_")[1]
    annotations, summary_stats, current_LD_block_positions = load_annot_and_summary_stats(
                joinpath(annot_data_path, block, "variant_list_annotated_adult_fetal.bed"),
                joinpath(annot_data_path, block, gwas_file_name);
                min_MAF = min_MAF
            )
    
    
    SNPs_count = size(annotations, 1)
    @info "$(ltime()) Number of SNPs in block: $SNPs_count"
    if isfile(model_file)
        @load model_file model
    else
        @info "$(ltime()) $model_file not found, creating new model!"
        # File doesn't exist, create a new model
        K = size(annotations, 2)
        layer_1 = Dense(K => H, relu; init = Flux.glorot_normal(gain = 0.0005))
        layer_output = Dense(H => 2)
        model = Chain(layer_1, layer_output)
    end

    mkpath(joinpath("data", block))
    LD_reference_filtered = joinpath("data", block, "filtered_EUR_current_block")
    snpdata = SnpData(joinpath(ld_panel_path, block, "filtered_EUR"))
    SnpArrays.filter(snpdata; des=LD_reference_filtered, f_snp = x -> x[:position] in current_LD_block_positions)
    LD_reference_filtered = LD_reference_filtered * ".bed"
    LD, D, good_variants = compute_LD(LD_reference_filtered)
    @assert !any(isnan.(LD))

    PRS = train_until_convergence(
        summary_stats.BETA[good_variants],
        summary_stats.SE[good_variants],
        LD, # correlation matrix already filtered for good variants
        D, # already filtered for good variants
        annotations[good_variants, :],
        model = model,
        N = summary_stats.N[good_variants],
        train_nn = train_nn
    )

    open(betas_output_file, "w") do io
        write(io, "variant\tmu\talpha\tvar\tss_beta\n")
        writedlm(io, [summary_stats.SNP[good_variants] PRS[1] PRS[2] PRS[3] summary_stats.BETA[good_variants]], "\t")
    end

    if train_nn
        model = PRS[4]
        @save model_file model
    end

    effects = interpret_model(
        block, 
        model_file, 
        annot_data_path, 
        gwas_file_name, 
        interpretation_output_file;
        min_MAF = min_MAF
    )
end


function interpret_model(block = "chr1_16103_1170341", model_file = "/data/abattle4/jweins17/PRS_runner/output/chr4/trained_model.bson", annot_data_path = "/data/abattle4/april/hi_julia/annotations/ccre/celltypes", gwas_file_name = "bmi_gwas.tsv", output_file = "effects.tsv"; min_MAF = 0.01)

    annot_file = joinpath(annot_data_path, block, "variant_list_annotated_adult_fetal.bed")
    annot = CSV.read(annot_file, DataFrame)
    #cell_types = names(annot)[5:226]
    cell_types = ["A_Cardiomyocyte", "Acinar", "Adipocyte", "Airway_Goblet", "Alpha_1", "Alpha_2", "Alveolar_Cap_Endo", "Alveolar_Type_1", "Alveolar_Type_2", "Alveolar_Type_2_Immune", "Astrocyte_1", "Astrocyte_2", "BBB_Endothelial", "Beta_1", "Beta_2", "Cardiac_Fibroblast", "Cardiac_Pericyte_1", "Cardiac_Pericyte_2", "Cardiac_Pericyte_3", "Cardiac_Pericyte_4", "Chief", "Cilliated", "Club", "Colon_Epithelial_1", "Colon_Epithelial_2", "Colon_Epithelial_3", "Colon_Goblet", "Cortical_Epithelial", "Delta+Gamma", "Ductal", "Endocardial", "Endothelial_Exocrine", "Endothelial_General_1", "Endothelial_General_2", "Endothelial_General_3", "Endothelial_Myocardial", "Enteric_Neuron", "Enterochromaffin", "Enterocyte", "Esophageal_Epithelial", "Fasciculata", "Fibro_Epithelial", "Fibro_General", "Fibro_GI", "Fibro_Liver_Adrenal", "Fibro_Muscle", "Fibro_Nerve", "Follicular", "Foveolar", "GABA_1", "GABA_2", "Gastric_Neuroendocrine", "Glomerulosa", "Glutamatergic_1", "Glutamatergic_2", "Hepatocyte", "Keratinocyte_1", "Keratinocyte_2", "Luteal", "Lymphatic", "Macrophage_General", "Macrophage_Gen_or_Alv", "Mammary_Basal_Epi", "Mammary_Epithelial", "Mammary_Luminal_Epi_1", "Mammary_Luminal_Epi_2", "Mast", "Melanocyte", "Memory_B", "Mesothelial", "Microglia", "Myoepithelial", "Naive_T", "Natural_Killer_T", "Nerve_Stromal", "Oligodendrocyte", "Oligo_Precursor", "Paneth", "Parietal", "Pericyte_General_1", "Pericyte_General_2", "Pericyte_General_3", "Pericyte_General_4", "Pericyte_Muscularis", "Plasma_B", "Satellite", "Schwann_General", "SI_Goblet", "Skin_Basal_Epidermal", "Skin_Eccrine_Epidermal", "Skin_Granular_Epidermal", "Sm_Ms_Colon_1", "Sm_Ms_Colon_2", "Sm_Ms_GE_junction", "Sm_Ms_General", "Sm_Ms_GI", "Sm_Ms_Mucosal", "Sm_Ms_Muscularis_1", "Sm_Ms_Muscularis_2", "Sm_Ms_Muscularis_3", "Sm_Ms_Uterine", "Sm_Ms_Vaginal", "T_Lymphocyte_1_CD8+", "T_lymphocyte_2_CD4+", "Transitional_Cortical", "Tuft", "Type_II_Skeletal_Myocyte", "Type_I_Skeletal_Myocyte", "Vasc_Sm_Muscle_1", "Vasc_Sm_Muscle_2", "V_Cardiomyocyte", "Fetal_A_Cardiomyocyte", "Fetal_Acinar_1", "Fetal_Acinar_2", "Fetal_Adrenal_Cortical", "Fetal_Adrenal_Neuron", "Fetal_Alveolar_Cap_Endo", "Fetal_Alveolar_Epithelial_1", "Fetal_Alveolar_Epithelial_2", "Fetal_Astrocyte_1", "Fetal_Astrocyte_2", "Fetal_Astrocyte_3", "Fetal_Astrocyte_4", "Fetal_Astrocyte_5", "Fetal_B_Lymphocyte_1_SPIB+", "Fetal_B_Lymphocyte_2_CXCR5+", "Fetal_B_Lymphocyte_3_NPY+", "Fetal_Cardiac_Fibroblast", "Fetal_Cholangiocyte", "Fetal_Chromaffin", "Fetal_Cilliated", "Fetal_Ductal", "Fetal_Endocardial", "Fetal_Endothelial_General_1", "Fetal_Endothelial_General_2", "Fetal_Endothelial_General_3", "Fetal_Endothelial_Hepatic_1", "Fetal_Endothelial_Hepatic_2", "Fetal_Endothelial_Placental", "Fetal_Enteric_Glia", "Fetal_Enteric_Neuron", "Fetal_Enterocyte_1", "Fetal_Enterocyte_2", "Fetal_Enterocyte_3", "Fetal_Enteroendocrine", "Fetal_Erythroblast_1", "Fetal_Erythroblast_2", "Fetal_Erythroblast_3", "Fetal_Erythroblast_4", "Fetal_Erythroblast_5", "Fetal_Excitatory_Neuron_10", "Fetal_Excitatory_Neuron_11", "Fetal_Excitatory_Neuron_12", "Fetal_Excitatory_Neuron_1", "Fetal_Excitatory_Neuron_2", "Fetal_Excitatory_Neuron_3", "Fetal_Excitatory_Neuron_4", "Fetal_Excitatory_Neuron_5", "Fetal_Excitatory_Neuron_6", "Fetal_Excitatory_Neuron_7", "Fetal_Excitatory_Neuron_8", "Fetal_Excitatory_Neuron_9", "Fetal_Extravillous_Trophoblast", "Fetal_Fibro_General_1", "Fetal_Fibro_General_2", "Fetal_Fibro_General_3", "Fetal_Fibro_General_4", "Fetal_Fibro_General_5", "Fetal_Fibro_GI", "Fetal_Fibro_Muscle_1", "Fetal_Fibro_Placental_1", "Fetal_Fibro_Placental_2", "Fetal_Fibro_Splenic", "Fetal_Gastri_Goblet", "Fetal_Goblet", "Fetal_Hematopoeitic_Stem", "Fetal_Hepatoblast", "Fetal_Inhibitory_Neuron_1", "Fetal_Inhibitory_Neuron_2", "Fetal_Inhibitory_Neuron_3", "Fetal_Inhibitory_Neuron_4", "Fetal_Inhibitory_Neuron_5", "Fetal_Islet", "Fetal_Lymphatic", "Fetal_Macrophage_General_1", "Fetal_Macrophage_General_2", "Fetal_Macrophage_General_3", "Fetal_Macrophage_General_4", "Fetal_Macrophage_Hepatic_1", "Fetal_Macrophage_Hepatic_2", "Fetal_Macrophage_Hepatic_3", "Fetal_Macrophage_Placental", "Fetal_Megakaryocyte", "Fetal_Mesangial_1", "Fetal_Mesangial_2", "Fetal_Mesothelial", "Fetal_Metanephric", "Fetal_Oligo_Progenitor_2", "Fetal_Parietal+Chief", "Fetal_Photoreceptor", "Fetal_Placental_Neuron", "Fetal_Pulmonary_Neuroendocrine", "Fetal_Retinal_Neuron", "Fetal_Retinal_Pigment", "Fetal_Retinal_Progenitor_1", "Fetal_Retinal_Progenitor_2", "Fetal_Satellite_1", "Fetal_Satellite_2", "Fetal_Schwann_General", "Fetal_Skeletal_Myocyte_1", "Fetal_Skeletal_Myocyte_2", "Fetal_Skeletal_Myocyte_3", "Fetal_Stellate", "Fetal_Sympathoblast", "Fetal_Syncitio+Cytotrophoblast", "Fetal_Thymocyte", "Fetal_T_Lymphocyte_1_CD4+", "Fetal_T_Lymphocyte_2_Cytotoxic", "Fetal_T_Lymphocyte_3_IL2+", "Fetal_T_Lymphocyte_4_FASLG+", "Fetal_Ureteric_Bud", "Fetal_V_Cardiomyocyte", "AF_ALT"]
    rename!(annot,:snp_id => :SNP)
    annotations, summary_stats, current_LD_block_positions = load_annot_and_summary_stats(
                annot_file,
                joinpath(annot_data_path, block, gwas_file_name);
                min_MAF = min_MAF
            )

    @load model_file model
    nn_σ2_β, nn_p_causal = predict_with_nn(model, annotations)

    max_p_causal_index = argmax(nn_p_causal)
    annotations_copy = copy(annotations)
    effects_σ2_β = zeros(length(cell_types))
    effects_p_causal = zeros(length(cell_types))

    for j in eachindex(cell_types)
        annotations_copy[:, j] .= 1
        nn_σ2_β_j, nn_p_causal_j = predict_with_nn(model, Float32.(annotations_copy))
        effects_σ2_β[j] = nn_σ2_β_j[max_p_causal_index] - nn_σ2_β[max_p_causal_index]
        effects_p_causal[j] = nn_p_causal_j[max_p_causal_index] - nn_p_causal[max_p_causal_index]
        annotations_copy[:, j] = annotations[:, j] # reset the j-th column to the original values
    end

    df = DataFrame(cell_type = cell_types, effects_σ2_β = effects_σ2_β, effects_p_causal = effects_p_causal)

    open(output_file, "w") do io
        write(io, "cell_type\teffects_variance\teffects_PIP\n")
        writedlm(io, [df.cell_type df.effects_σ2_β df.effects_p_causal], "\t")
    end

    return df
end
