rule figure1:
    input: 
        "figures/figure1.png",

rule figure2:
    input: 
        "figures/figure2.png",

rule figure3:
    input: 
        "figures/seqdeft_visualization.svg",
        "figures/seqdeft_vcregression_comparison.svg",
        "figures/vcregression_visualization.svg",
        "figures/mut_eff_posterior.svg",

rule figure4:
    input: 
        "figures/thermodynamic_model_params.svg",
        "figures/thermodynamic_model_visualization.svg",
        "figures/thermodynamic_model_visualization_energies.svg",

rule figureS1:
    input:
        "figures/times_rendering.svg",

rule figureS2:
    input:
        "figures/times_laplacian_operator.svg",

rule figureS3:
    input:
        "figures/times_P_U_operator.svg",

rule figureS4:
    input:
        "figures/mave_predictions.svg",

rule figureS5:
    input:
        "figures/e_coli.seqdeft_visualization_axis3.png",

rule figureS6:
    input: 
        "figures/b_sub.5utr_sequence_logo.svg",
        "figures/b_sub.seqdeft_visualization.svg",
        "figures/b_sub.seqdeft.contrasts.svg",
        "figures/seqdeft_species_comparison.svg",

rule figureS7:
    input:
        "figures/vcregression_visualization_mfs.png",

rule figureS8:
    input:
        "figures/vcregression_visualization_axes.png",

rule figureS9:
    input:
        "figures/td_fit.png",

rule figureS10:
    input:
        "figures/rna_model_pred.svg",
        "figures/rna_model_visualization.svg",

rule main_figures:
    input:
        rules.figure1.input,
        rules.figure2.input,
        rules.figure3.input,
        rules.figure4.input,

rule supplementary_figures:
    input:
        rules.figureS1.input,
        rules.figureS2.input,
        rules.figureS3.input,
        rules.figureS4.input,
        rules.figureS5.input,
        rules.figureS6.input,
        rules.figureS8.input,
        rules.figureS9.input,
        rules.figureS10.input,

rule all:
    input:
        rules.main_figures.input,
        rules.supplementary_figures.input

rule prep_mave_data:
    input: 
        "data/dmsc.csv"
    output: 
        "processed/dmsc.csv",
        "processed/dmsc.train.csv",
        "processed/dmsc.test.csv",
    conda: 
        "sd"
    shell: 
        "python scripts/prep_data/get_mave_data.py"

rule get_edges:
    output: 
        "results/edges.npz",
    conda: 
        "sd"
    shell: 
        "python scripts/prep_data/get_edges.py"

rule prep_sequences:
    input: 
        "data/Escherichia_coli_gca_001263735.ASM126373v1.dna.toplevel.fa",
        "data/Escherichia_coli_gca_001263735.ASM126373v1.51.gtf",
        "data/GCF_000009045.1_ASM904v1_genomic.fa",
        "data/GCF_000009045.1_ASM904v1_genomic.gff",
    output: 
        "processed/e_coli.gene_5utr.csv",
        "processed/b_sub.gene_5utr.csv",
    conda: 
        "sd"
    shell: 
        "python scripts/prep_data/get_5utr_sequence_data.py"

rule prep_sd_sequences:
    input:
        "processed/e_coli.gene_5utr.csv",
        "processed/b_sub.gene_5utr.csv",
    output:
        "processed/e_coli.seqs.txt",
        "processed/b_sub.seqs.txt",
    conda: 
        "sd"
    shell:
        "python scripts/prep_data/get_SD_sequence_data.py"


rule mei:
    input: 
        "processed/dmsc.train.csv",
        "processed/dmsc.test.csv"
    output:
        "results/mei.map.csv",
        "results/mei.test_pred.csv",
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/models/mei_predict.py"

rule vc_fit:
    input: 
        "processed/dmsc.train.csv",
        "processed/dmsc.test.csv"
    output:
        "results/vcregression.lambdas.npy",
        "results/vcregression.variance_components.csv",
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/models/vc_fit.py"

rule vc_predict:
    input: 
        "processed/dmsc.train.csv",
        "processed/dmsc.test.csv",
        "results/vcregression.lambdas.npy"
    output:
        "results/vcregression.map.csv",
        "results/vcregression.test_pred.csv",
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/models/vc_predict.py"

rule vc_contrast:
    input: 
        "processed/dmsc.train.csv",
        "processed/dmsc.test.csv",
        "results/vcregression.lambdas.npy",
    output:
        "results/vcregression.contrasts.csv"
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/models/vc_contrasts.py"

rule vc_calc_visualization:
    input: 
        "results/vcregression.map.csv",
    output:
        "results/vcregression.map.mf_1.0.nodes.pq",
        "results/vcregression.map.mf_1.5.nodes.pq",
        "results/vcregression.map.mf_2.0.nodes.pq",
        "results/vcregression.map.mf_2.5.nodes.pq"
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/prep_results/vc_calc_visualization.py"

rule td_fit:
    input: 
        "processed/dmsc.train.csv",
        "processed/dmsc.test.csv"
    output:
        "results/thermodynamic_model.pth",
        "results/thermodynamic_model.ll.csv",
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/models/td_fit.py"

rule td_predict:
    input: 
        "processed/dmsc.train.csv",
        "processed/dmsc.test.csv",
        "results/thermodynamic_model.pth"
    output:
        "results/thermodynamic_model.pred.csv"
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/models/td_predict.py"

rule td_calc_visualization:
    input: 
        "results/thermodynamic_model.pred.csv",
    output:
        "results/thermodynamic_model.nodes.pq"
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/prep_results/td_calc_visualization.py"

rule rna_calc_energies:
    output:
        "processed/rna_model.energies.csv"
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/models/rna_calc_energies.py"

rule rna_fit:
    input:
        "processed/rna_model.energies.csv",
        "processed/dmsc.train.csv",
        "processed/dmsc.test.csv"
    output:
        "results/rna_model.pth",
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/models/rna_fit.py"

rule rna_predict:
    input:
        "processed/rna_model.energies.csv",
        "processed/dmsc.train.csv",
        "processed/dmsc.test.csv",
        "results/rna_model.pth"
    output:
        "results/rna_model.pred.csv"
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/models/rna_predict.py"

rule rna_calc_visualization:
    input:
        "results/rna_model.pred.csv",
    output:
        "results/rna_model.nodes.pq"
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/prep_results/rna_calc_visualization.py"

rule calc_r2_curves:
    input:
        "processed/dmsc.csv",
    output:
        "results/models.r2.csv"
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/models/mei_vc_r2_curves.py"

rule seqdeft_fit:
    input: 
        "processed/e_coli.seqs.csv",
        "processed/b_sub.seqs.csv",
    output:
        "results/e_coli.seqdeft.a.npy",
        "results/b_sub.seqdeft.a.npy",
        "results/e_coli.seqdeft.cv_results.csv",
        "results/b_sub.seqdeft.cv_results.csv",
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/models/seqdeft_fit.py"

rule seqdeft_predict:
    input: 
        "processed/e_coli.seqs.csv",
        "processed/b_sub.seqs.csv",
        "results/e_coli.seqdeft.a.npy",
        "results/b_sub.seqdeft.a.npy",
    output:
        "results/e_coli.seqdeft.map.csv",
        "results/b_sub.seqdeft.map.csv",
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/models/seqdeft_predict.py"

rule seqdeft_contrast:
    input: 
        "processed/e_coli.seqs.csv",
        "processed/b_sub.seqs.csv",
        "results/e_coli.seqdeft.a.npy",
        "results/b_sub.seqdeft.a.npy",
    output:
        "results/e_coli.seqdeft.contrasts.csv",
        "results/b_sub.seqdeft.contrasts.csv",
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/models/seqdeft_contrasts.py"

rule seqdeft_calc_visualization:
    input: 
        "results/e_coli.seqdeft.map.csv",
        "results/b_sub.seqdeft.map.csv",
    output:
        "results/e_coli.seqdeft.map.nodes.pq",
        "results/b_sub.seqdeft.map.nodes.pq",
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/prep_results/seqdeft_calc_visualization.py"

rule benchmark_laplacian_operator:
    output:
        "results/times_laplacian_operator.csv",
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/prep_results/calc_laplacian_times.py"

rule benchmark_P_U_operator:
    output:
        "results/times_P_U_operator.csv",
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/prep_results/calc_P_U_times.py"

rule benchmark_rendering_times:
    output:
        "results/times_rendering.csv",
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/prep_results/calc_rendering_times.py"

rule calc_variance_components:
    input:
        "results/vcregression.map.csv",
    output:
        "results/vcregression.map.variance_components.csv",
        "results/vcregression.map.site_marginal_epistasis.csv",
        "results/vcregression.map.pairwise_marginal_epistasis.csv",
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/prep_results/calc_map_variance_components.py"

# Making plots
rule plot_figure1:
    input:
        "processed/e_coli.gene_5utr.csv",
        "processed/e_coli.seqs.csv",
        "processed/e_coli.seqdeft.map.csv",
        "processed/e_coli.seqdeft.cv_results.csv",
    output:
        "figures/figure1.png"
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figure1/figure1.py"

rule plot_figure2:
    input:
        "results/dmsc.empirical_distance_correlation.csv",
        "results/vc.prior_variance_components.csv",
        "results/vcregression.map_variance_components.csv",
        "processed/dmsc.train.csv",
        "processed/dmsc.test.csv",
        "results/vcregression.map.csv",
        "results/vcregression.test_pred.csv",
        "results/vcregression.map.site_marginal_epistasis.csv",
        "results/vcregression.map_pairwise_marginal_epistasis.csv",
    output:
        "figures/figure2.png"
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figure2/figure2.py"

rule plot_seqdeft_visualization:
    input:
        "results/e_coli.seqdeft.map.nodes.pq",
        "results/e_coli.seqdeft.map.decay_rates.csv",
        "results/edges.npz",
    output:
        "figures/seqdeft_visualization.svg"
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figure3/plot_seqdeft_visualization.py"

rule plot_seqdeft_vcregression_comparison:
    input:
        "results/e_coli.seqdeft.map.csv",
        "results/vcregression.map.csv",
    output:
        "figures/seqdeft_vcregression_comparison.svg"
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figure3/plot_data_comparison.py"

rule plot_vcregression_visualization:
    input:
        "results/vcregression.map.mf_2.nodes.pq",
        "results/edges.npz",
    output:
        "figures/vcregression_visualization.svg"
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figure3/plot_vcregression_visualization.py"

rule plot_contrasts:
    input:
        "results/vcregression.contrasts.csv",
        "results/e_coli.seqdeft.contrasts.csv",
    output:
        "figures/mut_eff_posterior.svg"
    conda: 
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figure3/plot_contrasts.py"

rule plot_td_params:
    input:
        "results/thermodynamic_model.pth",
    output:
        "figures/thermodynamic_model_params.svg"
    conda:
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figure4/plot_td_params.py"

rule plot_td_visualization:
    input:
        "results/thermodynamic_model.nodes.pq",
        "results/edges.npz",
    output:
        "figures/thermodynamic_model_visualization.svg"
    conda:
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figure4/plot_td_visualization.py"

rule plot_td_visualization_energies:
    input:
        "results/thermodynamic_model.nodes.pq",
        "results/edges.npz",
    output:
        "figures/thermodynamic_model_visualization_energies.svg"
    conda:
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figure4/plot_td_visualization_energies.py"

rule plot_b_sub_contrast:
    input:
        "results/b_sub.seqdeft.contrasts.csv",
    output:
        "figures/b_sub.seqdeft.contrasts.png",
        "figures/b_sub.seqdeft.contrasts.svg"
    conda:
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figures_supp/plot_b_sub_contrasts.py"
  
rule plot_b_sub_sequence_logo:
    input:
        "processed/b_sub.gene_5utr.csv",
    output:
        "figures/b_sub.5utr_sequence_logo.png",
        "figures/b_sub.5utr_sequence_logo.svg",
    conda:
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figures_supp/plot_b_sub_sequence_logo.py"
  
rule plot_b_sub_visualization:
    input:
        "results/b_sub.seqdeft.nodes.pq",
        "results/edges.npz"
    output:
        "figures/b_sub.seqdeft_visualization.png",
        "figures/b_sub.seqdeft_visualization.svg",
    conda:
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figures_supp/plot_b_sub_sequence_logo.py"

rule plot_seqdeft_species_comparison:
    input:
        "results/e_coli.seqdeft.map.csv",
        "results/b_sub.seqdeft.map.csv",
    output:
        "figures/seqdeft_species_comparison.png",
        "figures/seqdeft_species_comparison.svg",
    conda:
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figures_supp/plot_seqdeft_species_comparison.py"

rule plot_rendering_benchmark:
    input:
        "results/times_rendering.csv",
    output:
        "figures/times_rendering.png",
        "figures/times_rendering.svg",
    conda:
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figures_supp/plot_rendering_times.py"
  
rule plot_laplacian_benchmark:
    input:
        "results/times_laplacian_operator.csv",
    output:
        "figures/times_laplacian_operator.png",
        "figures/times_laplacian_operator.svg",
    conda:
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figures_supp/plot_laplacian_benchmark.py"

rule plot_P_U_benchmark:
    input:
        "results/times_P_U_operator.csv",
    output:
        "figures/times_P_U_operator.png",
        "figures/times_P_U_operator.svg",
    conda:
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figures_supp/plot_P_U_benchmark.py"

rule plot_mave_predictions:
    input:
        "results/r2.csv",
        "processed/dmsc.test.csv",
        "results/vcregression.test_pred.csv",
        "results/mei.test.csv",
    output:
        "figures/mave_predictions.png",
        "figures/mave_predictions.svg",
    conda:
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figures_supp/plot_mave_predictions.py"
  
rule plot_seqdeft_visualization_axes:
    input:
        "results/e_coli.seqdeft.map.nodes.pq",
        "results/edges.npz",
    output:
        "figures/e_coli.seqdeft_visualization_axis3.png",
    conda:
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figures_supp/plot_seqdeft_visualization_axes.py"

rule plot_vcregression_visualization_mean_functions:
    input:
        "results/vcregression.map.mf_1.nodes.pq",
        "results/vcregression.map.mf_1.5.nodes.pq",
        "results/vcregression.map.mf_2.nodes.pq",
        "results/vcregression.map.mf_2.5.nodes.pq",
        "results/edges.npz",
    output:
        "figures/vcregression_visualization_mfs.png",
    conda:
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figures_supp/plot_vcregression_visualizations_mf.py"

rule plot_vcregression_visualization_axes:
    input:
        "results/vcregression.map.mf_2.nodes.pq",
        "results/edges.npz",
    output:
        "figures/vcregression_visualization_axes.png",
    conda:
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figures_supp/plot_vcregression_visualizations_axes.py"

rule plot_td_fit:
    input:
        "results/thermodynamic_model.pred.csv",
        "processed/dmsc.train.csv",
        "processed/dmsc.test.csv",
        "results/thermodynamic_model.ll.csv",
    output:
        "figures/td_fit.png",
        "figures/td_fit.svg",
    conda:
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figures_supp/plot_td_fit.py"

rule plot_rna_model_predictions:
    input:
        "results/rna_model.pred.csv",
        "processed/dmsc.train.csv",
        "processed/dmsc.test.csv",
    output:
        "figures/rna_model_pred.png",
        "figures/rna_model_pred.svg",
    conda:
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figures_supp/plot_rna_model_predictions.py"

rule plot_rna_model_visualization:
    input:
        "results/rna_model.nodes.pq",
        "results/edges.npz",
    output:
        "figures/rna_model_visualization.png",
        "figures/rna_model_visualization.svg",
    conda:
        "sd"
    shell:
        "source activate.sh ; python scripts/figures/figures_supp/plot_rna_model_visualization.py"
